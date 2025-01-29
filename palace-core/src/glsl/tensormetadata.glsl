#define TensorMetaDataI(N) TensorMetaDataImpl ## N
#define TensorMetaData(N) TensorMetaDataI(N)

#define _N 1
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 2
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 3
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 4
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 5
#include <tensormetadata_generic.glsl>
#undef _N
