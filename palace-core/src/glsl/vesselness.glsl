float sato_vesselness(float l1, float l2, float l3, float two_alpha1_squared, float two_alpha2_squared) {
    // Do a quick bubble sort so that l1 >= l2 >= l3
    if(l1 < l2) {
        swap(l1, l2);
    }
    if(l2 < l3) {
        swap(l2, l3);
    }
    if(l1 < l2) {
        swap(l1, l2);
    }

    //float lc = min(-l2, -l3);
    float lc = -l2;
    if(lc == 0) {
        return 0;
    }

    if(l1 <= 0) {
        return exp((-l1*l1)/(two_alpha1_squared*lc*lc))*lc;
    } else {
        return exp((-l1*l1)/(two_alpha2_squared*lc*lc))*lc;
    }
}
