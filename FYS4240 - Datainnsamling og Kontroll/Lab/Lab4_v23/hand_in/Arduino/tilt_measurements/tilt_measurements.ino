/*
  Arduino BMI270 - Simple Accelerometer

  This example reads the acceleration values from the BMI270
  sensor and continuously prints them to the Serial Monitor
  or Serial Plotter.

  The circuit:
  - Arduino Nano 33 BLE Sense Rev2

  created 10 Jul 2019
  by Riccardo Rizzo

  This example code is in the public domain.
*/

#include "Arduino_BMI270_BMM150.h"
#include "math.h"

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Acceleration in G's");
  Serial.println("X\tY\tZ");
  delay(500);     //To see what is printed in the setup
}

//Converting accelerometer output to a defined body frame to get a right-handed coordinate system:
//Set G_px=-x, G_py=-y and G_pz=z

//Both functions return values in radians. Convert them to angles using identity(180/PI)

//atan2f(float) returns an angle in correct quadrant
float calc_roll_angle(float y, float z){
  return atan2f(-y,z)*(180/PI);    
}

float calc_pitch_angle(float x, float y, float z){
  return atan(x/sqrt(sq(-y)+sq(z)))*(180/PI);
}

void loop() {
  float x, y, z;
  float roll_angle = calc_roll_angle(y,z);      //phi
  float pitch_angle = calc_pitch_angle(x,y,z);  //theta
  float yaw_angle = 0.0;    // Yaw must be set to zero when board is static

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);

    //Serial.print(x);
    //Serial.print('\t');
    //Serial.print(y);
    //Serial.print('\t');
    //Serial.println(z);
    Serial.print(roll_angle);
    Serial.print('\t');
    Serial.print(pitch_angle);
    Serial.print('\t');
    Serial.print(yaw_angle);
    Serial.println('\r\n');
    delay(10);          //gives stability to the readings in labView but decreases the loop frequency
    }
}
*/