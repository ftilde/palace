// Polyfill because some graphics cards (AMD) don't supply atomic add for float
// TODO: Use the proper instructions if available.
#define atomic_add_float(mem, value) {\
    uint initial = 0;\
    uint new = 0;\
    do {\
        initial = mem;\
        new = floatBitsToUint(uintBitsToFloat(initial) + (value));\
        if (new == initial) {\
            break;\
        }\
    } while(atomicCompSwap(mem, initial, new) != initial);\
}
