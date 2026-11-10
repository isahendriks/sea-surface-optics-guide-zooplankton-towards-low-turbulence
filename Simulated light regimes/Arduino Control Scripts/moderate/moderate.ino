#include <RGBmatrixPanel.h>

#define CLK  8  

#define OE   9
#define LAT 10
#define A   A0
#define B   A1
#define C   A2
#define D   A3

RGBmatrixPanel matrix(A, B, C, D, CLK, LAT, OE, false);

void setup() {
  matrix.begin();
}


void loop() {

  int i, interval, bar_width, bar_height, shifts, intensity;

  interval = 20;
  bar_width = 1;
  bar_height = 32;
  intensity = 30; 

  shifts = 32 / bar_width; // for a 32 x 32 LED Matrix
  

  for (i = 0; i < shifts + 1; i++) {
      
     matrix.fillRect(i * bar_width, 0, bar_width, bar_height, matrix.Color888(intensity, intensity, intensity));
     delay(interval);
     matrix.fillRect(0, 0, 32, 32, matrix.Color888(0, 0, 0));
     
  }
 
}
