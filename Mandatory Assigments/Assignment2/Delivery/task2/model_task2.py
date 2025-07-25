import torch
from torch import nn


class ImageCaptionModel(nn.Module):
    def __init__(
        self,
        cnn_feature_dim,
        embed_size,
        hidden_size,
        vocab_size,
        max_caption_length,
        num_layers,
        cell_type,
        use_attention=False,
    ):
        """
        The main Image Captioning Model class.
        :param cnn_feature_dim (int): Dimensionality of the frozen image features from ResNet.
        :param embed_size (int): Dimensionality of the word embeddings.
        :param hidden_size (int): Hidden state size of the RNN/LSTM cell.
        :param vocab_size (int): Size of the vocabulary.
        :param max_caption_length (int): Maximum caption length.
        :param num_layers (int): Number of layers of RNN/LSTM cell.
        :param cell_type (str): 'RNN' or 'LSTM'.
        :param use_attention (bool): Use the attention mechanism if True
        """
        super(ImageCaptionModel, self).__init__()
        self.use_attention = use_attention
        self.feature_projection = nn.Sequential(
            nn.Dropout(p=0.25), nn.Linear(cnn_feature_dim, hidden_size), nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.caption_rnn = CaptionRNN(
            cnn_feature_dim=cnn_feature_dim,
            vocabulary_size=vocab_size,
            embedding_size=embed_size,
            hidden_state_size=hidden_size,
            max_caption_length=max_caption_length,
            num_layers=num_layers,
            cell_type=cell_type,
            use_attention=use_attention,
        )

    def forward(self, cnn_features, token_ids, is_train):
        """
        :param cnn_features (torch.Tensor): Features from the CNN; shape [batch, num_regions, feat_dim].
        :param token_ids (torch.Tensor): Token indices, shape [batch, seq_len]. The 1st token should be the start token.
        :param is_train (bool): If True, use teacher forcing; otherwise, use inference mode.
        :return: logits (torch.Tensor): Logits for each time step. Shape: [seq_len, batch, vocabulary_size].
                attn_weights (List): Returned only if use_attention is True. A list of attention map weights for each
                                     time step. Shape: [seq_len, batch, num_regions]
        """
        if self.use_attention:
            # TODO: Permute features: from [batch, channels, H, W] to [batch, H, W, channels],
            #       then flatten spatial dimensions to get [batch, num_regions, channels].
            cnn_features = None
            # TODO: Apply the projection to each region (nn.Linear applies to the last dim).
            processed_img_feat = (
                None  # Resulting shape should be [batch, num_regions, hidden_size]
            )
            logits, attn_weights = self.caption_rnn(
                tokens=token_ids, features=processed_img_feat, is_train=is_train
            )
            return logits, attn_weights
        else:
            # Convert features from shape [batch_size, 512, 7, 7] to [batch_size, 512]
            cnn_features = self.avgpool(cnn_features).squeeze(-1).squeeze(-1)
            processed_img_feat = self.feature_projection(
                cnn_features
            )  # Resulting shape: [batch_size, hidden_size]
            logits, final_hidden = self.caption_rnn(
                tokens=token_ids, features=processed_img_feat, is_train=is_train
            )
            return logits, None


class CaptionRNN(nn.Module):
    def __init__(
        self,
        cnn_feature_dim,
        vocabulary_size,
        embedding_size,
        hidden_state_size,
        max_caption_length,
        num_layers,
        cell_type,
        use_attention=False,
    ):
        """
        Combined RNN with optional attention mechanism.

        Args:
            cnn_feature_dim (int): Dimensionality of the (possibly projected) feature vector(s).
            vocabulary_size (int): Vocabulary size.
            embedding_size (int): Dimensionality of token embeddings.
            hidden_state_size (int): Hidden state size of the RNN.
            max_caption_length (int): Maximum caption length.
            num_layers (int): Number of RNN layers.
            cell_type (str): 'RNN' or 'LSTM'.
            use_attention (bool): If True, use attention mechanism over features.
        """
        super(CaptionRNN, self).__init__()
        self.use_attention = use_attention
        self.hidden_state_size = hidden_state_size
        self.num_layers = num_layers
        self.max_caption_length = max_caption_length
        self.vocabulary_size = vocabulary_size

        # Embedding and output layers.
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        # The output layer is used to convert the hidden state to vocabulary size.
        self.output_layer = nn.Linear(self.hidden_state_size, self.vocabulary_size)

        # Create attention module if needed.
        if self.use_attention:
            self.attention = Attention(
                cnn_feature_dim=cnn_feature_dim, hidden_dim=hidden_state_size
            )
        input_sizes = []
        for i in range(num_layers):
            if i == 0:
                # First layer takes concatenated embedding and feature vector
                input_sizes.append(embedding_size + hidden_state_size)
            else:
                # Other layers take the /hidden state size of the previous layer
                input_sizes.append(hidden_state_size)

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if cell_type == "RNN":
                self.cells.append(RNNCell(input_sizes[i], hidden_state_size))
            elif cell_type == "LSTM":
                self.cells.append(LSTMCell(input_sizes[i], hidden_state_size))
            else:
                raise ValueError("cell_type should be either 'RNN' or 'LSTM'")

    def forward(self, tokens, features, is_train):
        """
        Forward pass for the caption RNN.

        Args:
            tokens (Tensor): Token indices, shape [batch, seq_len].
            features (Tensor):
                If use_attention is True, expected shape [batch, num_regions, feature_dim];
                Otherwise, shape [batch, feature_dim].
            is_train (bool): If True, use teacher forcing.

        Returns:
            logits (Tensor): Logits for each time step, shape [seq_len, batch, vocabulary_size].
            attn_weights (list[Tensor] or None): List of attention weights per time step if using attention;
                                                 otherwise, None. Shape: [seq_len, batch, num_regions]
        """
        batch_size, seq_len = tokens.size()
        if not is_train:
            seq_len = self.max_caption_length

        # Embed all tokens (for teacher forcing).
        token_embeddings = self.embedding(tokens)  # [batch, seq_len, embedding_size]

        # TODO: Initialize hidden_states with correct dimensions depending on the cell type.
        # hidden_states is a list of length self.num_layers with each element having a tensor of zeros of shape
        # (batch_size, 2 * self.hidden_state_size).
        # We use (2 * self.hidden_state_size) because we need a hidden state AND a cell memory for LSTM.
        # We do not need this size for vanilla RNN cell but we modify the RNNCell instead to have the same
        # interface for both RNNCell and LSTMCell. This avoids putting if statements at some places in the
        # code below.
        # Initialize hidden_states with zeros
        # Each hidden state has shape (batch_size, 2 * hidden_state_size)
        # For RNN, the second half is unused but we maintain the same interface
        hidden_states = [
            torch.zeros(batch_size, 2 * self.hidden_state_size, device=tokens.device)
            for _ in range(self.num_layers)
        ]

        logits_series = []
        attn_weights_series = [] if self.use_attention else None

        # TODO: Fetch the first (index 0) embeddings that should go as input to the RNN.
        # Get first token embedding (start token)
        current_token_vec = token_embeddings[
            :, 0, :
        ]  # Shape (batch_size, embedding_size)

        for t in range(seq_len):
            new_states = []
            # TODO:
            # 1. Loop over the RNN layers and provide them with correct input. Inputs depend on the layer
            #    index so input for layer-0 will be different from the input for other layers.
            # 2. Update the hidden cell state for every layer.
            # 3. If you are at the last layer, then produce logits_i, predictions. Append logits_i to logits_series.
            for layer in range(self.num_layers):
                if layer == 0:
                    if self.use_attention:
                        # TODO:  Compute attention: use previous hidden state (only the first half of
                        #        hidden_states variable) to attend over features.
                        prev_hidden = None
                        context, alpha = self.attention(features, prev_hidden)
                        attn_weights_series.append(alpha)
                        # TODO: Concatenate token embedding with the attended image context.
                        cell_input = None
                    else:
                        # TODO: Without attention, concatenate the token embedding with the image feature.
                        cell_input = torch.cat([current_token_vec, features], dim=1)
                else:
                    # For higher layers, input is the hidden state from the previous layer
                    # We only use the first half of the hidden state
                    cell_input = hidden_states[layer - 1][:, : self.hidden_state_size]
                    # TODO: Initialise to the output of the previous layer

                # call cell for this layer
                new_state = self.cells[layer](cell_input, hidden_states[layer])
                # TODO: Call the cell for this layer
                new_states.append(new_state)

                if layer == self.num_layers - 1:
                    # TODO: Get logits from the self.output_layer and append them to logits_series
                    hidden = new_state[:, : self.hidden_state_size]
                    logits = self.output_layer(
                        hidden
                    )  # Shape: [batch, vocabulary_size]
                    logits_series.append(logits)

                    # for next time step, we need to get the token embedding for the next token
                    if t < seq_len - 1:
                        if is_train:
                            current_token_vec = token_embeddings[:, t + 1, :]
                        else:
                            predicted_tokens = torch.argmax(logits, dim=1)
                            current_token_vec = self.embedding(predicted_tokens)
            hidden_states = new_states

        logits = torch.stack(logits_series, dim=1)  # Convert to a tensor
        return logits, attn_weights_series


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_state_size):
        super(RNNCell, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_state_size)
        self.hidden_to_hidden = nn.Linear(hidden_state_size, hidden_state_size)

    def forward(self, x, state_old):
        # state_old: [batch, 2 * hidden_state_size]; we use only the first half.
        h_old = state_old[:, : state_old.size(1) // 2]
        h_new = torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(h_old))
        # Concatenate new hidden state with itself to mimic (hidden_state, cell_memory) for LSTM-like interface.
        return torch.cat([h_new, h_new], dim=1)


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_state_size: int):
        """
        :param input_size: Size (number of units/features) of the input to LSTM
        :param hidden_state_size: Size (number of units/features) in the hidden state of LSTM
        """
        super(LSTMCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        # TODO: Initialise weights and biases for the forget gate (weight_f, bias_f), input gate (w_i, b_i),
        #       output gate (w_o, b_o), and hidden state (weight, bias)
        #       self.weight, self.weight_(f, i, o):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(f, i, o): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to two.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n
        #       Note: The actual input tensor will have 2 * HIDDEN_STATE_SIZE because it contains both
        #             hidden state and cell's memory

        # Forget gate parameters - Initialize weights and biases for forget gate
        self.weight_f = nn.Parameter(
            torch.Tensor(input_size + hidden_state_size, hidden_state_size)
        )
        self.bias_f = nn.Parameter(torch.Tensor(1, hidden_state_size))

        # Input gate parameters - Initialize weights and biases for the input gate
        self.weight_i = nn.Parameter(
            torch.Tensor(input_size + hidden_state_size, hidden_state_size)
        )
        self.bias_i = nn.Parameter(torch.Tensor(1, hidden_state_size))

        # Output gate parameters - Initialize weights and biases for the output gate
        self.weight_o = nn.Parameter(
            torch.Tensor(input_size + hidden_state_size, hidden_state_size)
        )
        self.bias_o = nn.Parameter(torch.Tensor(1, hidden_state_size))

        # Memory cell parameters
        self.weight = nn.Parameter(
            torch.Tensor(input_size + hidden_state_size, hidden_state_size)
        )
        self.bias = nn.Parameter(torch.Tensor(1, hidden_state_size))

        # Initialize weights and biases w/ variance scaling
        for weight in [
            self.weight_f,
            self.weight_i,
            self.weight_o,
            self.weight,
        ]:
            nn.init.xavier_normal_(weight)
        # Init biases - set forget gate bias to 1 to help with long-term dependencies
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.bias_i)
        nn.init.zeros_(self.bias_o)
        nn.init.ones_(self.bias_f)  # Set forget bias to 1 as recommended

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for an LSTM unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the LSTM.
        :param hidden_state: A tensor with shape [batch_size, 2 * HIDDEN_STATE_SIZE] containing the hidden
                             state and the cell memory. The 1st half represents the hidden state and the
                             2nd half represents the cell's memory
        :return: The updated hidden state (including memory) of the LSTM cell.
                 Shape: [batch_size, 2 * HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the LSTM equations to get the new hidden state, cell memory and return them.
        #       The first half of the returned value must represent the new hidden state and the second half
        #       new cell state.
        # Split the hidden state into h and c components
        h_prev = hidden_state[:, : self.hidden_state_size]
        c_prev = hidden_state[:, self.hidden_state_size :]

        # Concatenate the input and the previous hidden state
        combined = torch.cat((x, h_prev), dim=1)

        # Compute the forget gate
        f_t = torch.sigmoid(torch.matmul(combined, self.weight_f) + self.bias_f)

        # Compute the input gate
        i_t = torch.sigmoid(torch.matmul(combined, self.weight_i) + self.bias_i)

        # Compute the output gate
        o_t = torch.sigmoid(torch.matmul(combined, self.weight_o) + self.bias_o)

        # Compute the candidate memory cell
        c_tilde = torch.tanh(torch.matmul(combined, self.weight) + self.bias)

        # Update the cell state
        c_t = f_t * c_prev + i_t * c_tilde

        # Compute the new hidden state
        h_t = o_t * torch.tanh(c_t)

        # Concatenate h_t and c_t to create the new hidden state
        new_hidden_state = torch.cat([h_t, c_t], dim=1)

        return new_hidden_state


class Attention(nn.Module):
    def __init__(self, cnn_feature_dim, hidden_dim):
        """
        A simple attention module.
        Args:
            cnn_feature_dim (int): Number of channels in the encoder feature maps.
            hidden_dim (int): Dimensionality of the hidden state.
        """
        super(Attention, self).__init__()
        # TODO: Create layers to project the hidden state and the image features to a common dimension.
        self.hidden_to_hidden = (
            None  # Linear layer to project hidden_state to hidden_dim
        )
        self.features_to_hidden = (
            None  # Linear layer to project cnn_feature_dim to hidden_dim
        )

        self.hidden_to_attention_score = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, hidden_state):
        """
        Args:
            features (Tensor): Encoded image features,
                                  shape: [batch, num_regions, encoder_dim].
            hidden_state (Tensor): Hidden state from the LSTM,
                                   shape: [batch, hidden_dim].
        Returns:
            context (Tensor): Weighted image feature vector, shape: [batch, encoder_dim].
            alpha (Tensor): Attention weights, shape: [batch, num_regions].
        """
        # TODO: Project hidden state and unsqueeze() to [batch, 1, hidden_dim] so that it can be broadcast.
        hidden_proj = None
        # TODO: Project encoder outputs
        enc_proj = None  # Resulting Shape: [batch, num_regions, hidden_dim]
        # TODO: Add the projections and apply self.relu
        att = None
        # TODO: Get scalar attention scores for each region of the image
        att_scores = None  # Resulting shape: [batch, num_regions]
        # TODO: Apply softmax on the att_scores to get the alphas (attention weights)
        alpha = None
        # TODO: Compute context vector as the weighted sum of encoder outputs. You might need to unsqueeze() alpha.
        context = None
        return context, alpha
