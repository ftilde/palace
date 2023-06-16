#include <util.glsl>

struct SymMat3 {
    float xx;
    float xy;
    float xz;
    float yz;
    float yy;
    float zz;
};

//TODO: not sure if we need double precision here
#define complex vec2

complex c_mul(complex a, complex b) {
    return complex(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

complex c_mul3(complex a, complex b, complex c) {
    return c_mul(c_mul(a, b), c);
}
complex c_div(complex a, complex b) {
    return complex(((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y)),((a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y)));
}

complex c_sqrt(complex a) {
    float l = length(a);
    float real = sqrt(0.5*(l+a.x));
    float imag = sqrt(0.5*(l-a.x));
    if (a.y < 0.0) {
        imag = -imag;
    }
    return complex(real, imag);
}

vec2 c_pow(complex a, float n) {
    float angle = atan(a.y, a.x);
    float r = length(a);
    float real = pow(r, n) * cos(n*angle);
    float imag = pow(r, n) * sin(n*angle);
    return complex(real, imag);
}

void eigenvalues_impl(float xx, float xy, float xz, float yy, float yz, float zz, out float l1, out float l2, out float l3) {
    // Set up characteristic equation:   det( A - lambda I ) = 0
    //    as a cubic in lambda:  a.lambda^3 + b.lambda^2 + c.lambda + d = 0
    complex a = complex(-1.0);                // -1
    complex b = complex(xx + yy + zz);        // trace
    complex c = complex(yz * yz - yy * zz     // -sum of diagonal minors
                      + xz * xz - zz * xx
                      + xy * xy - xx * yy);
    complex d = complex(xx*yy*zz + 2*xy*yz*xz - xx*yz*yz - yy*xz*xz - zz*xy*xy);

    // Solve cubic by Cardano's method (easier in complex numbers!)
    complex p = c_div(c_mul(b, b) - 3.0 * c_mul(a, c), 9.0 * c_mul(a, a));
    complex q = c_div(9.0 * c_mul3(a, b, c) - 27.0 * c_mul3(a, a, d) - 2.0 * c_mul3(b, b, b), 54.0 * c_mul3(a, a, a));
    complex delta = c_mul(q, q) - c_mul3(p, p, p);
    complex deltaSqrt = c_sqrt(delta);
    complex g1 = c_pow(q + deltaSqrt, 1.0 / 3.0);
    complex g2 = c_pow(q - deltaSqrt, 1.0 / 3.0);
    complex offset = c_div(-b, 3.0 * a);
    complex omega = complex(-0.5, 0.5 * sqrt( 3.0 ));
    complex omega2 = c_mul(omega, omega);

    complex cl1 = g1                + g2                + offset;
    complex cl2 = c_mul(g1, omega)  + c_mul(g2, omega2) + offset;
    complex cl3 = c_mul(g1, omega2) + c_mul(g2, omega)  + offset;

    l1=cl1.x;
    l2=cl2.x;
    l3=cl3.x;
}

void eigenvalues(SymMat3 m, out float l1, out float l2, out float l3) {
    eigenvalues_impl(m.xx, m.xy, m.xz, m.yy, m.yz, m.zz, l1, l2, l3);
}
