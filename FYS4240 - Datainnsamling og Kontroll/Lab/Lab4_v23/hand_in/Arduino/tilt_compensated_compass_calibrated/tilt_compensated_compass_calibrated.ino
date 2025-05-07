/*
  Arduino BMM150 - Simple Magnetometer

  This example reads the magnetic field values from the BMM150
  sensor and continuously prints them to the Serial Monitor
  or Serial Plotter.

  The circuit:
  - Arduino Nano 33 BLE Sense Rev2

  created 10 Jul 2019
  by Riccardo Rizzo

  This example code is in the public domain.
*/



#include "Arduino_BMI270_BMM150.h"

#define DECLINATION (4.47)
#define DEG_TO_RAD (PI/180.0)
#define RAD_TO_DEG (180.0/PI)

//atan2f(float) returns an angle in correct quadrant
float calc_roll_angle(float y, float z){              //finds the roll_anlge
  return atan2f(-y,z )* RAD_TO_DEG;    
}

float calc_pitch_angle(float x, float y, float z){    //find the pitch angle
  return atan(x/sqrt(sq(-y)+sq(z))) * RAD_TO_DEG;
}

//A part of Exercise 4.5 taken from Arduino Cookbook


float constrainAngle360(float dta) {
 dta = fmod(dta, 2.0 * PI);
 if (dta < 0.0)
  dta += 2.0 * PI;
 return dta * RAD_TO_DEG;
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.print("Magnetic field sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.print(IMU.magneticFieldSampleRate());
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Magnetic Field in uT");
  Serial.println("Pitch\tRoll\tYaw");
  delay(500);
}

void loop() {
  float x_m, y_m, z_m;    //readings from magnetometer
  float x_a, y_a, z_a;    //readings from acceleratormeter
  

  if (IMU.magneticFieldAvailable() && IMU.accelerationAvailable()) {
    IMU.readAcceleration(x_a,y_a,z_a);
    IMU.readMagneticField(x_m, y_m, z_m);
     

    //apply transformation matrix R^(BM)
    float B_x = -x_m;
    float B_y = y_m;
    float B_z = -z_m; 
    

    //Compute pitch and roll angles. Negative z-comp. as we convert it to frame B
    float theta = calc_pitch_angle(x_a,y_a,z_a);  //pitch
    float phi = calc_roll_angle(y_a,z_a);         //roll
  
    //Tilt compensation with offset correction
    float B_hx = B_x * cos(theta) + B_y * sin(phi) * sin(theta) - B_z * cos(phi) * sin(theta) + 18.0570; //offsets for delta_B_x
    float B_hy = B_y * cos(phi) + B_z * sin(phi) + 3.2854;                                       //offsets for delta_B_y
    
    //Yaw w/ declination correction
    float psi = constrainAngle360(atan2f(B_hy, B_hx) + (DECLINATION * DEG_TO_RAD));
    //float psi = atan2(B_hy,B_hx)*(180/PI);
    //float psi = psi_constrained * RAD_TO_DEG;

    //Serial.print(B_x);
    Serial.print(theta);
    Serial.print('\t');
    //Serial.print(B_y);
    Serial.print(phi);
    //Serial.print(B_z);
    Serial.print('\t');
    Serial.println(psi);
    
  }
}
