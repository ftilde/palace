uint[_N] add(uint[_N] l, uint[_N] r) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] + r[i];
    }
    return res;
}

uint[_N] sub(uint[_N] l, uint[_N] r) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] - r[i];
    }
    return res;
}

uint[_N] saturating_sub(uint[_N] l, uint[_N] r) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] > r[i] ? l[i] - r[i] : 0;
    }
    return res;
}

uint[_N] mul(uint[_N] l, uint[_N] r) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] * r[i];
    }
    return res;
}

uint[_N] div(uint[_N] l, uint[_N] r) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] / r[i];
    }
    return res;
}

uint[_N] min(uint[_N] l, uint[_N] r) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = min(l[i], r[i]);
    }
    return res;
}

uint[_N] max(uint[_N] l, uint[_N] r) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = max(l[i], r[i]);
    }
    return res;
}

float[_N] add(float[_N] l, float[_N] r) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] + r[i];
    }
    return res;
}

float[_N] sub(float[_N] l, float[_N] r) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] - r[i];
    }
    return res;
}

float[_N] mul(float[_N] l, float[_N] r) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] * r[i];
    }
    return res;
}

float[_N] scale(float[_N] l, float a) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] * a;
    }
    return res;
}

float[_N] neg(float[_N] l) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = -l[i];
    }
    return res;
}

float[_N] div(float[_N] l, float[_N] r) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] / r[i];
    }
    return res;
}

float[_N] mod(float[_N] l, float[_N] r) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        float div = l[i] / r[i];
        res[i] = fract(div) * r[i];
    }
    return res;
}

float[_N] min(float[_N] l, float[_N] r) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = min(l[i], r[i]);
    }
    return res;
}

float[_N] max(float[_N] l, float[_N] r) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = max(l[i], r[i]);
    }
    return res;
}

int[_N] add(int[_N] l, int[_N] r) {
    int[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] + r[i];
    }
    return res;
}

int[_N] sub(int[_N] l, int[_N] r) {
    int[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] - r[i];
    }
    return res;
}

int[_N] mul(int[_N] l, int[_N] r) {
    int[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] * r[i];
    }
    return res;
}

int[_N] div(int[_N] l, int[_N] r) {
    int[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] / r[i];
    }
    return res;
}

int[_N] min(int[_N] l, int[_N] r) {
    int[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = min(l[i], r[i]);
    }
    return res;
}

int[_N] max(int[_N] l, int[_N] r) {
    int[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = max(l[i], r[i]);
    }
    return res;
}

float[_N] clamp(float[_N] val, float[_N] low, float[_N] high) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = clamp(val[i], low[i], high[i]);
    }
    return res;
}

float[_N] abs(float[_N] l) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = abs(l[i]);
    }
    return res;
}

uint[_N] fill(uint[_N] dummy, uint val) {
    uint[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = val;
    }
    return res;
}
int[_N] fill(int[_N] dummy, int val) {
    int[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = val;
    }
    return res;
}
float[_N] fill(float[_N] dummy, float val) {
    float[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = val;
    }
    return res;
}

uint[_N] div_round_up(uint[_N] v1, uint[_N] v2) {
    return div(sub(add(v1, v2), fill(v1, 1)), v2);
}

uint[_N] from_linear(uint linear_pos, uint[_N] dim) {
    uint[_N] res;
    for(int i = _N-1; i>= 0; i-=1) {
        uint ddim = dim[i];
        res[i] = linear_pos % ddim;
        linear_pos /= ddim;
    }

    // In case of overflow return a value that should definitely be picked up
    // by other checks
    if(linear_pos != 0) {
        return fill(res, 0xffffffff);
    }

    return res;
}

uint to_linear(uint[_N] pos, uint[_N] dim) {
    uint res = pos[0];
    for(int i=1; i<_N; i+=1) {
        res = res * dim[i] + pos[i];
    }
    return res;
}

uint64_t to_linear64(uint[_N] pos, uint[_N] dim) {
    uint64_t res = uint64_t(pos[0]);
    for(int i=1; i<_N; i+=1) {
        res = res * uint64_t(dim[i]) + uint64_t(pos[i]);
    }
    return res;
}

bool[_N] less_than(int[_N] l, int[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] < r[i];
    }
    return res;
}
bool[_N] less_than(uint[_N] l, uint[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] < r[i];
    }
    return res;
}
bool[_N] less_than(float[_N] l, float[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] < r[i];
    }
    return res;
}

bool[_N] less_than_equal(uint[_N] l, uint[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] <= r[i];
    }
    return res;
}

bool[_N] less_than_equal(int[_N] l, int[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] <= r[i];
    }
    return res;
}

bool[_N] less_than_equal(float[_N] l, float[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] <= r[i];
    }
    return res;
}
bool[_N] equal(int[_N] l, int[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] == r[i];
    }
    return res;
}

bool[_N] equal(uint[_N] l, uint[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+=1) {
        res[i] = l[i] == r[i];
    }
    return res;
}

bool all(bool[_N] v) {
    bool res = true;
    for(int i=0; i<_N; i+=1) {
        res = res && v[i];
    }
    return res;
}

bool any(bool[_N] v) {
    bool res = false;
    for(int i=0; i<_N; i+=1) {
        res = res || v[i];
    }
    return res;
}

bool[_N] and(bool[_N] l, bool[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = l[i] && r[i];
    }
    return res;
}

bool[_N] or(bool[_N] l, bool[_N] r) {
    bool[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = l[i] || r[i];
    }
    return res;
}

uint hmul(uint[_N] v) {
    uint res = 1;
    for(int i=0; i<_N; i+=1) {
        res *= v[i];
    }
    return res;
}

float hmin(float[_N] v) {
    float res = v[0];
    for(int i=1; i<_N; i+=1) {
        res = min(res, v[i]);
    }
    return res;
}
float hmax(float[_N] v) {
    float res = v[0];
    for(int i=1; i<_N; i+=1) {
        res = max(res, v[i]);
    }
    return res;
}

float dot(float[_N] l, float[_N] r) {
    float res=0.0;
    for(int i=0; i<_N; i+= 1) {
        res += l[i]*r[i];
    }
    return res;
}

float[_N+1] to_homogeneous(float[_N] v) {
    float[_N+1] res;
    for(int i=0; i<_N; i+= 1) {
        res[i+1] = v[i];
    }
    res[0] = 1.0;
    return res;
}

float[_N] from_homogeneous(float[_N+1] v) {
    float[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = v[i+1];
    }
    return res;
}

float[_N] to_float(uint[_N] v) {
    float[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = float(v[i]);
    }
    return res;
}

int[_N] to_int(float[_N] v) {
    int[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = int(v[i]);
    }
    return res;
}

int[_N] to_int(uint[_N] v) {
    int[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = int(v[i]);
    }
    return res;
}

uint[_N] to_uint(int[_N] v) {
    uint[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = uint(v[i]);
    }
    return res;
}

uint[_N] to_uint(float[_N] v) {
    uint[_N] res;
    for(int i=0; i<_N; i+= 1) {
        res[i] = uint(v[i]);
    }
    return res;
}
