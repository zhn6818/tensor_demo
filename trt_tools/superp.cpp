#include "superp.h"

SuperPointC::SuperPointC(Config *config) : trtInterface(config)
{
    trtInterface::initDims();
}

// bool SuperPointC::build(bool isAddOptiProfile)
// {
//     return trtInterface::build(isAddOptiProfile);
// }