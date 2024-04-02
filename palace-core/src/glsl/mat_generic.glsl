float[_N] mul(Mat(_N) m, float[_N] v) {
    float[_N] res;
    for(int or=0; or<_N; or+= 1) {
        float r = 0.0;
        for(int i=0; i<_N; i+= 1) {
            r += m[i][or] * v[i];
        }
        res[or] = r;
    }
    return res;
}
