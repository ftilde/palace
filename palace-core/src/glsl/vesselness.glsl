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

float sq(float v) {
    return v*v;
}

float frangi_vesselness(float l1, float l2, float l3, float alpha, float beta, float c) {
    // Do a quick bubble sort so that |l1| <= |l2| <= |l3|
    if(abs(l1) > abs(l2)) {
        swap(l1, l2);
    }
    if(abs(l2) > abs(l3)) {
        swap(l2, l3);
    }
    if(abs(l1) > abs(l2)) {
        swap(l1, l2);
    }

    float rb = abs(l1) / sqrt(abs(l2 * l3));
    float ra = abs(l2 / l3);
    float s = sqrt(sq(l1) + sq(l2) + sq(l3));

    if(l2 > 0 || l3 > 0) {
        return 0;
    } else {
        return (1.0 - exp(-sq(ra)/(2*sq(alpha)))) * exp(-sq(rb)/(2*sq(beta))) * (1.0 - exp(-sq(s)/(2*sq(c))));
    }
}
