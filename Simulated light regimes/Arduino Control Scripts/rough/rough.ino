// Central circle representing moon, with its radius and intensity changing.
// Random circles emerging here and there represenging random distortions.

#include <RGBmatrixPanel.h>
#include <math.h>

#define CLK 8
#define OE 9
#define LAT 10
#define A A0
#define B A1
#define C A2
#define D A3

RGBmatrixPanel matrix(A, B, C, D, CLK, LAT, OE, false);

void setup()
{
    matrix.begin();
    randomSeed(analogRead(0)); // Initialize random seed
}

void loop()
{
    int center_x, center_y, max_radius, min_radius, intensity_main, intensity, interval, i;

    center_x = 16;   // x coordinate of center of circle
    center_y = 16;   // y coordinate of center of circle
    max_radius = 10; // initial radius of the moon
    min_radius = 5;  // final radius of the moon
    intensity_main = 55; // initial intensity of the moon initial 128 not below 50 (0-255)
    interval = 100; //higher value slower, lower faster

    // Simulate waves passing with changing radius and intensity
    for (i = min_radius; i < max_radius; i++)
    {
        // Calculate intensity based on the radius and a sine function
        intensity = intensity_main * sin((i - min_radius) * 3.14 / 10);

        // Draw the moon with varying radius and intensity
        fillCircle(matrix, center_x, center_y, i, matrix.Color888(intensity, intensity, intensity));

        // Add random distortions in the matrix
        addRandomDistortions();

        delay(interval);
        matrix.fillScreen(matrix.Color888(0, 0, 0)); // Clear the screen
    }

    // Simulate waves receding
    for (i = max_radius - 1; i > min_radius - 1; i--)
    {
        // Calculate intensity based on the radius and a sine function
        intensity = intensity_main * sin((i - min_radius) * 3.14 / 10);

        // Draw the moon with varying radius and intensity
        fillCircle(matrix, center_x, center_y, i, matrix.Color888(intensity, intensity, intensity));

        // Add random distortions in the matrix
        addRandomDistortions();

        delay(interval);
        matrix.fillScreen(matrix.Color888(0, 0, 0)); // Clear the screen
    }
}

void fillCircle(RGBmatrixPanel &matrix, int x0, int y0, int r, uint16_t color)
{
    for (int y = -r; y <= r; y++)
    {
        for (int x = -r; x <= r; x++)
        {
            if (x * x + y * y <= r * r)
            {
                matrix.drawPixel(x0 + x, y0 + y, color);
            }
        }
    }
}

void addRandomDistortions()
{
    for (int j = 0; j < 10; j++)
    {
        int x = random(32);             // Random x coordinate
        int y = random(32);             // Random y coordinate
        int size = random(2, 3);        // Random size for circles or rectangles
        int intensity = random(10, 20); // Random intensity

        if (random(2) == 0)
        {
            // Draw a circle
            fillCircle(matrix, x, y, size, matrix.Color888(intensity, intensity, intensity));
        }
        else
        {
            // Draw a rectangle
            matrix.fillRect(x, y, size, size, matrix.Color888(intensity, intensity, intensity));
        }
    }
}
