#!/usr/bin/env python3
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Your specific file path
FILE_PATH = r"C:\Users\danfr\Documents\Studie\V25\FYS4240 - Datainnsamling og Kontroll\Lab\lab5_v25\Example Binary File.dat"


def analyze_binary_file(file_path):
    """
    Comprehensive analysis of a binary file with unknown format.
    Attempts multiple interpretations to help identify the data format.
    """
    print(f"Analyzing binary file: {file_path}")

    # Verify the file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        print("Please check that the path is correct and the file exists.")
        return None

    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")

    # Open the file in binary mode
    with open(file_path, "rb") as f:
        raw_data = f.read()

    # Basic statistics
    print("\n=== BASIC BYTE STATISTICS ===")
    # Count unique bytes and show distribution of the top 10
    byte_counts = Counter(raw_data)
    print(f"Number of unique bytes: {len(byte_counts)}")
    print("Top 10 most common bytes:")
    for byte, count in byte_counts.most_common(10):
        print(f"0x{byte:02x}: {count} times ({count/file_size*100:.2f}%)")

    # Check if file size is divisible by common data type sizes
    print("\n=== FILE SIZE ANALYSIS ===")
    for dtype_name, dtype_size in [
        ("8-bit (U8/I8)", 1),
        ("16-bit (U16/I16)", 2),
        ("32-bit (U32/I32/float)", 4),
        ("64-bit (U64/I64/double)", 8),
    ]:
        if file_size % dtype_size == 0:
            print(
                f"File size is divisible by {dtype_size} bytes ({dtype_name}): {file_size // dtype_size} elements"
            )
        else:
            remainder = file_size % dtype_size
            print(
                f"Not cleanly divisible by {dtype_size} bytes, remainder: {remainder} bytes"
            )

    # Try different interpretations
    print("\n=== DATA INTERPRETATION ===")

    # Define how many elements to sample for each data type
    sample_size = min(20, file_size // 8)  # Limit sample size

    # Function to safely interpret data with different formats
    def interpret_data(format_str, bytes_per_value, endianness_name):
        try:
            format_size = struct.calcsize(format_str)
            element_count = file_size // format_size

            if element_count == 0:
                return None

            # Sample the beginning of the file
            sample_count = min(sample_size, element_count)
            values = list(struct.unpack_from(format_str * sample_count, raw_data))

            # Basic statistics on the sample
            if values:
                return {
                    "values": values,
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "element_count": element_count,
                }
            return None
        except Exception as e:
            print(f"Error interpreting as {format_str} ({endianness_name}): {str(e)}")
            return None

    # Try different formats and endianness
    formats = [
        ("<f", 4, "Little-endian float (32-bit)"),
        (">f", 4, "Big-endian float (32-bit)"),
        ("<d", 8, "Little-endian double (64-bit)"),
        (">d", 8, "Big-endian double (64-bit)"),
        ("<h", 2, "Little-endian short (16-bit)"),
        (">h", 2, "Big-endian short (16-bit)"),
        ("<i", 4, "Little-endian int (32-bit)"),
        (">i", 4, "Big-endian int (32-bit)"),
        ("<q", 8, "Little-endian long (64-bit)"),
        (">q", 8, "Big-endian long (64-bit)"),
        ("B", 1, "Unsigned char (8-bit)"),
    ]

    results = {}
    for fmt, size, name in formats:
        if file_size >= size:  # Only if file is big enough
            result = interpret_data(fmt, size, name)
            if result:
                results[name] = result
                print(f"\n{name}:")
                print(
                    f"  Sample values: {[round(v, 4) if isinstance(v, float) else v for v in result['values'][:5]]}"
                )
                print(f"  Range: {result['min']} to {result['max']}")
                print(f"  Mean: {result['mean']}")
                print(f"  Est. element count: {result['element_count']}")

    # Plot the most promising interpretations
    print("\n=== VISUALIZATION ===")
    plot_interpretations(results, raw_data, file_path)

    return results


def plot_interpretations(results, raw_data, file_path):
    """Plot the data with the most likely interpretations"""

    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(file_path), "binary_analysis_results")
    os.makedirs(output_dir, exist_ok=True)

    # Base filename for outputs
    base_filename = os.path.basename(file_path).replace(".", "_")

    # Prioritize plotting based on data type likelihood
    # 1. Check if there are any float/double with reasonable values
    float_types = [
        "Little-endian float (32-bit)",
        "Big-endian float (32-bit)",
        "Little-endian double (64-bit)",
        "Big-endian double (64-bit)",
    ]

    # Check if any of the float types have reasonable values
    reasonable_float_types = []
    for ftype in float_types:
        if ftype in results:
            # Check if values are in a reasonable range and not all identical
            values = results[ftype]["values"]
            if (
                -1e10 < results[ftype]["min"] < 1e10
                and -1e10 < results[ftype]["max"] < 1e10
            ):
                if results[ftype]["max"] - results[ftype]["min"] > 1e-10:
                    reasonable_float_types.append(ftype)

    # If no reasonable float interpretations, try integers
    if not reasonable_float_types:
        # Try plotting integer types
        int_types = [
            "Little-endian short (16-bit)",
            "Big-endian short (16-bit)",
            "Little-endian int (32-bit)",
            "Big-endian int (32-bit)",
            "Little-endian long (64-bit)",
            "Big-endian long (64-bit)",
        ]
        for itype in int_types:
            if itype in results:
                plot_data(
                    results[itype]["values"],
                    itype,
                    os.path.join(
                        output_dir,
                        f"{base_filename}_{itype.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.png",
                    ),
                )
    else:
        # Plot all reasonable float types
        for ftype in reasonable_float_types:
            plot_data(
                results[ftype]["values"],
                ftype,
                os.path.join(
                    output_dir,
                    f"{base_filename}_{ftype.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.png",
                ),
            )

    # Plot byte frequency
    plt.figure(figsize=(10, 6))
    byte_counts = Counter(raw_data)
    bytes_to_plot = sorted(byte_counts.keys())
    counts_to_plot = [byte_counts[b] for b in bytes_to_plot]

    plt.bar([f"0x{b:02x}" for b in bytes_to_plot[:40]], counts_to_plot[:40])
    plt.title("Byte Frequency Distribution (First 40 unique bytes)")
    plt.xlabel("Byte Value (hex)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_byte_dist.png"))
    plt.close()

    print(f"Saved plots to {output_dir}")


def plot_data(values, interpretation, output_path):
    """Plot the interpreted data"""
    plt.figure(figsize=(12, 6))

    # Plot only the first 1000 values if there are more
    length = min(1000, len(values))
    x = np.arange(length)
    y = values[:length]

    plt.plot(x, y)
    plt.title(f"Data interpreted as {interpretation}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)

    # Add some statistical information
    plt.figtext(
        0.02,
        0.02,
        f"Min: {min(y):.4g}, Max: {max(y):.4g}, Mean: {sum(y)/len(y):.4g}",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def try_numpy_import(file_path):
    """Try to import the file using numpy with different data types"""
    print("\n=== NUMPY IMPORT ATTEMPTS ===")
    dtypes = [
        ("f4", "Little-endian float (32-bit)"),
        (">f4", "Big-endian float (32-bit)"),
        ("f8", "Little-endian double (64-bit)"),
        (">f8", "Big-endian double (64-bit)"),
        ("i2", "Little-endian short (16-bit)"),
        (">i2", "Big-endian short (16-bit)"),
        ("i4", "Little-endian int (32-bit)"),
        (">i4", "Big-endian int (32-bit)"),
    ]

    results = {}
    for dtype_str, name in dtypes:
        try:
            data = np.fromfile(file_path, dtype=np.dtype(dtype_str))
            if len(data) > 0:
                # Check if the values are reasonable
                if np.isfinite(data).all() and not np.isnan(data).any():
                    results[name] = {
                        "values": data[:20].tolist(),
                        "min": float(np.min(data)),
                        "max": float(np.max(data)),
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "element_count": len(data),
                    }
                    print(f"\n{name}:")
                    print(
                        f"  First few values: {[round(v, 4) if isinstance(v, float) else v for v in data[:5]]}"
                    )
                    print(f"  Range: {results[name]['min']} to {results[name]['max']}")
                    print(
                        f"  Mean: {results[name]['mean']}, Std: {results[name]['std']}"
                    )
                    print(f"  Element count: {results[name]['element_count']}")
        except Exception as e:
            print(f"Error reading with NumPy as {dtype_str}: {str(e)}")

    return results


def print_labview_recommendations(file_path, results):
    """Print recommendations for LabVIEW import settings"""
    file_size = os.path.getsize(file_path)

    print("\n=== RECOMMENDATIONS FOR LABVIEW ===")
    print("Based on the analysis, here are recommended settings for LabVIEW:")

    # Check for reasonable float interpretations first
    float_types = [
        "Little-endian double (64-bit)",
        "Big-endian double (64-bit)",
        "Little-endian float (32-bit)",
        "Big-endian float (32-bit)",
    ]

    # Find best interpretation
    best_type = None
    for ftype in float_types:
        if ftype in results:
            # Check if values are reasonable
            result = results[ftype]
            min_val, max_val = result["min"], result["max"]
            if -1e10 < min_val < 1e10 and -1e10 < max_val < 1e10:
                if max_val - min_val > 1e-10:  # Not all identical values
                    best_type = ftype
                    break

    # If no reasonable float type found, try integers
    if best_type is None:
        int_types = [
            "Little-endian int (32-bit)",
            "Big-endian int (32-bit)",
            "Little-endian short (16-bit)",
            "Big-endian short (16-bit)",
        ]
        for itype in int_types:
            if itype in results:
                best_type = itype
                break

    # Provide LabVIEW recommendations
    if best_type:
        print(f"\nBest interpretation appears to be: {best_type}")

        # Map to LabVIEW data types
        labview_type = None
        byte_order = None

        if "double" in best_type:
            labview_type = "DBL"
        elif "float" in best_type:
            labview_type = "SGL"
        elif "short" in best_type:
            labview_type = "I16"
        elif "int" in best_type:
            labview_type = "I32"

        if "Little-endian" in best_type:
            byte_order = "Little Endian"
        elif "Big-endian" in best_type:
            byte_order = "Big Endian"

        print(f"\nLabVIEW Read From Binary File settings:")
        print(f"1. Data type: {labview_type}")
        print(f"2. Byte order: {byte_order}")

        # Calculate element count
        if labview_type == "DBL":
            element_count = file_size // 8
        elif labview_type == "SGL":
            element_count = file_size // 4
        elif labview_type == "I32":
            element_count = file_size // 4
        elif labview_type == "I16":
            element_count = file_size // 2
        else:
            element_count = "Unknown"

        print(f"3. Count: {element_count}")

        # Sample values
        print("\nFirst few values to expect:")
        for i, val in enumerate(results[best_type]["values"][:5]):
            if isinstance(val, float):
                print(f"  {i}: {val:.6f}")
            else:
                print(f"  {i}: {val}")
    else:
        print("\nCould not determine the best interpretation.")
        print("Please review the analysis outputs manually.")


if __name__ == "__main__":
    print("Starting binary file analysis...")
    print(f"Target file: {FILE_PATH}")

    # Run the analysis
    results = analyze_binary_file(FILE_PATH)

    if results:
        # Also try NumPy import
        numpy_results = try_numpy_import(FILE_PATH)

        # Print LabVIEW recommendations
        print_labview_recommendations(FILE_PATH, results)

        print(
            "\nAnalysis complete. Check the generated plots for visual representations."
        )
