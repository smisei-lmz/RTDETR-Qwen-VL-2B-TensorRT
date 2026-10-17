#pragma once

#include <string>


struct Detection
{
    /* data */
    int class_id;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};


