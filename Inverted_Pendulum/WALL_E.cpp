#include <Wire.h>
#include <MPU6050.h>
#include "KalmanFilter.h"
#include <PinChangeInterrupt.h>
#include <util/atomic.h>

// Constants
#define NMOTORS 2
#define EMERGENCY_STOP_THRESHOLD 30.0  // Maximum allowable tilt angle (degrees) before stopping

// Motor parameters
const int enca[] = {4, 3};
const int encb[] = {5, 2};
const int pwm[] = {10, 11};
const int in1[] = {9, 6};
const int in2[] = {8, 7};

// LQR Controller Class
class LQR_Controller {
  private:
    double K_pitch, K_pitch_rate, K_theta_rate, umax;

  public:
    LQR_Controller() : K_pitch(0), K_pitch_rate(0), K_theta_rate(0), umax(255) {}

    void setParams(double kp, double kp_rate, double kt_rate, double umax_in) {
      K_pitch = kp;
      K_pitch_rate = kp_rate;
      K_theta_rate = kt_rate;
      umax = umax_in;
    }

    void computeControl(double theta_p, double theta_p_rate, double theta_w_rate, int &pwr, int &dir) {
      double u = -K_pitch * theta_p - K_pitch_rate * theta_p_rate - K_theta_rate * theta_w_rate;
      pwr = (int)fabs(u);
      if (pwr > umax) {
        pwr = umax;
      }
      dir = (u < 0) ? -1 : 1;
    }
};

// Global objects and variables
LQR_Controller lqr[NMOTORS];
MPU6050 mpu6050;
KalmanFilter kalmanfilter;
volatile int posi[] = {0, 0};
int16_t ax, ay, az, gx, gy, gz;
double K[] = {3.0826e3, 0.3333e3, 0.1000e3};  // Control gains
double loop_timer;
double x1, x2; // Angle states
double x3[] = {0.0, 0.0};
double v1Prev[NMOTORS] = {0.0, 0.0};
double velocity[NMOTORS] = {0.0, 0.0};
double v1[NMOTORS] = {0.0, 0.0};
double v1_filt[NMOTORS] = {0.0, 0.0};
int posPrev[NMOTORS] = {0, 0};
float dt = 0.01;
float Q_angle = 0.001, Q_gyro = 0.005, R_angle = 0.5, C_0 = 1, K1 = 0.05;

// Emergency stop function
void emergencyStop() {
  for (int k = 0; k < NMOTORS; k++) {
    analogWrite(pwm[k], 0);
    digitalWrite(in1[k], LOW);
    digitalWrite(in2[k], LOW);
  }
  Serial.println("Emergency Stop Activated!");
}

void setup() {
  Wire.begin();
  Serial.begin(9600);

  // Initialize MPU6050
  delay(1000);
  mpu6050.initialize();
  delay(4);

  // Initialize motors
  for (int k = 0; k < NMOTORS; k++) {
    pinMode(enca[k], INPUT);
    pinMode(encb[k], INPUT);
    pinMode(pwm[k], OUTPUT);
    pinMode(in1[k], OUTPUT);
    pinMode(in2[k], OUTPUT);
    lqr[k].setParams(K[0], K[1], K[2], 255);
  }

  attachPCINT(digitalPinToPCINT(enca[0]), readEncoder<0>, RISING);
  attachPCINT(digitalPinToPCINT(enca[1]), readEncoder<1>, RISING);

  loop_timer = micros() + 10000;
}

void loop() {
  mpu6050.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  kalmanfilter.Angle(ax, ay, az, gx, gy, gz, dt, Q_angle, Q_gyro, R_angle, C_0, K1);
  x1 = kalmanfilter.angle;
  x2 = kalmanfilter.angle_dot;

  // Safety check
  if (fabs(x1) > EMERGENCY_STOP_THRESHOLD) {
    emergencyStop();
    while (true);  // Halt system
  }

  for (int k = 0; k < NMOTORS; k++) {
    int pwr, dir;
    lqr[k].computeControl(x1, x2, x3[k], pwr, dir);
    setMotor(dir, pwr, pwm[k], in1[k], in2[k]);
  }

  // Update motor states
  int pos[NMOTORS];
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    for (int k = 0; k < NMOTORS; k++) {
      pos[k] = posi[k];
    }
  }

  for (int k = 0; k < NMOTORS; k++) {
    float deltaT = 10000 / 1.0e6;
    velocity[k] = ((double)(pos[k] - posPrev[k]) / deltaT);
    posPrev[k] = pos[k];
    v1[k] = ((float)(velocity[k] / 660 * 60)) * 6;
    v1_filt[k] = 0.854 * v1_filt[k] + 0.0728 * v1[k] + 0.0728 * v1Prev[k];
    v1Prev[k] = v1[k];
    x3[k] = v1_filt[k];
  }

  Serial.println(x2);
  while (loop_timer > micros());
  loop_timer += 10000;
}

void setMotor(int dir, int pwmVal, int pwm, int in1, int in2) {
  analogWrite(pwm, pwmVal);
  if (dir == 1) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  } else if (dir == -1) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  }
}

template <int o>
void readEncoder() {
  int b = digitalRead(encb[o]);
  if (b > 0) {
    posi[o]++;
  } else {
    posi[o]--;
  }
}
