// Polyfill because some graphics cards (AMD) don't supply atomic add for float
// TODO: Use the proper instructions if available.

#define atomic_combine_float(mem, value, combine) {\
    uint initial = 0;\
    uint new = 0;\
    do {\
        initial = mem;\
        new = floatBitsToUint(combine(uintBitsToFloat(initial), (value)));\
        if (new == initial) {\
            break;\
        }\
    } while(atomicCompSwap(mem, initial, new) != initial);\
}

#define _add(l, r) (l) + (r)
#define atomic_add_float(mem, value) atomic_combine_float(mem, value, _add)
#define atomic_max_float(mem, value) atomic_combine_float(mem, value, max)
#define atomic_min_float(mem, value) atomic_combine_float(mem, value, min)
