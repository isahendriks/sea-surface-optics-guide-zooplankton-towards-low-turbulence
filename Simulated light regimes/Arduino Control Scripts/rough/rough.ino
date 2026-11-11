// Rough light regime: a central disc representing the moon that grows and shrinks in radius
// and intensity, plus small randomly placed spots representing random distortions.

#include <RGBmatrixPanel.h>
#include <math.h>

// Pins of the LED matrix
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
    randomSeed(analogRead(0)); // random seed from an unconnected analog pin
}

void loop()
{
    int center_x = 16;       // x coordinate of the centre of the disc
    int center_y = 16;       // y coordinate of the centre of the disc
    int max_radius = 10;     // largest radius of the disc [LEDs]
    int min_radius = 5;      // smallest radius of the disc [LEDs]
    int intensity_main = 55; // peak intensity of the disc (0-255)
    int interval = 100;      // time per step [ms]

    // Wave passing: the disc grows, its intensity follows a sine of the radius
    for (int i = min_radius; i < max_radius; i++)
    {
        drawStep(center_x, center_y, i, min_radius, intensity_main, interval);
    }

    // Wave receding: the disc shrinks again
    for (int i = max_radius - 1; i > min_radius - 1; i--)
    {
        drawStep(center_x, center_y, i, min_radius, intensity_main, interval);
    }
}

// Draw the disc with the given radius and the random spots, wait, and clear the matrix
void drawStep(int center_x, int center_y, int radius, int min_radius, int intensity_main, int interval)
{
    int intensity = intensity_main * sin((radius - min_radius) * 3.14 / 10);
    fillCircle(matrix, center_x, center_y, radius, matrix.Color888(intensity, intensity, intensity));
    addRandomDistortions();

    delay(interval);
    matrix.fillScreen(matrix.Color888(0, 0, 0));
}

// Filled circle of radius r around (x0, y0)
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

// Ten spots at random positions: circles (radius 2) or squares (2 x 2 LEDs), intensity 10-19
void addRandomDistortions()
{
    for (int j = 0; j < 10; j++)
    {
        int x = random(32);
        int y = random(32);
        int size = random(2, 3);        // always 2 (the upper bound of random() is exclusive)
        int intensity = random(10, 20);

        if (random(2) == 0)
        {
            fillCircle(matrix, x, y, size, matrix.Color888(intensity, intensity, intensity));
        }
        else
        {
            matrix.fillRect(x, y, size, size, matrix.Color888(intensity, intensity, intensity));
        }
    }
}
