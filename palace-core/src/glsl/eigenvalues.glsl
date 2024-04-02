#include <util.glsl>

struct SymMat3 {
    float xx;
    float xy;
    float xz;
    float yz;
    float yy;
    float zz;
};

#define complex vec2
#define scalar float

complex c_mul(complex a, complex b) {
    return complex(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

complex c_sqrt(complex a) {
    scalar l = length(a);
    float u,v;
    if (a.x > 0.0) {
        u = sqrt(0.5*(l+a.x));
        v = a.y / (2.0*u);
    } else {
        v = sqrt(0.5*(l-a.x));
        u = a.y / (2.0*v);
    }
    return complex(u, v);
}

complex c_pow(complex a, float n) {
    float angle = atan(float(a.y), float(a.x));
    float r = float(length(a));
    float real = pow(r, n) * cos(n*angle);
    float imag = pow(r, n) * sin(n*angle);
    return complex(real, imag);
}

void eigenvalues_impl(float xx, float xy, float xz, float yy, float yz, float zz, out float l1, out float l2, out float l3) {
    // Set up characteristic polynomial:   det( A - lambda I ) = 0
    // The coefficients are:
    // a = -1
    // b = the trace
    // c = -sum of diagonal minors
    // d = the negative determinant
    scalar b = xx + yy + zz;
    scalar c = yz * yz - yy * zz + xz * xz - zz * xx + xy * xy - xx * yy;
    scalar d = xx*yy*zz + 2*xy*yz*xz - xx*yz*yz - yy*xz*xz - zz*xy*xy;

    // Solve cubic by Cardano's method (a is already factored into the
    // equations below as a constant).
    scalar p = (b * b + 3.0 * c) / 9.0;
    scalar q = (-9.0 * b * c - 27.0 * d - 2.0 * b * b * b) / -54.0;
    complex delta = complex(q*q - p*p*p, 0.0);
    complex q_c = complex(q, 0.0);
    complex deltaSqrt = c_sqrt(delta);
    complex g1 = c_pow(q_c + deltaSqrt, 1.0 / 3.0);
    complex g2 = c_pow(q_c - deltaSqrt, 1.0 / 3.0);
    scalar offset = b / 3.0;
    complex omega = complex(-0.5, 0.5 * sqrt( 3.0 ));
    complex omega2 = c_mul(omega, omega);

    scalar cl1 = g1.x                + g2.x                + offset;
    scalar cl2 = c_mul(g1, omega).x  + c_mul(g2, omega2).x + offset;
    scalar cl3 = c_mul(g1, omega2).x + c_mul(g2, omega).x  + offset;

    l1=float(cl1);
    l2=float(cl2);
    l3=float(cl3);
}

void eigenvalues(SymMat3 m, out float l1, out float l2, out float l3) {
    eigenvalues_impl(m.xx, m.xy, m.xz, m.yy, m.yz, m.zz, l1, l2, l3);
}
