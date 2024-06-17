#include <iostream>
#include <iomanip>
#include <boost/numeric/interval.hpp>


using namespace boost::numeric;

using namespace std;

int main(void) {

    float  piFloat  = 3.14159265358979323846;
    double piDouble = 3.14159265358979323846;

    interval<float> iFloat = interval<float>(-3.14159265358979323846, 3.14159265358979323846);
    interval<double> iDouble = interval<double>(-3.14159265358979323846, 3.14159265358979323846);

    interval<float> test = interval<float>(2,2);
    test /= 10;

    printf("\n\\****   Double vs Float   ****\\\n");

    cout << setprecision(64) << "Pi as entered:  3.14159265358979323846" << endl;
    cout << setprecision(64) << "Pi as double:   " << piDouble << endl;
    cout << setprecision(64) << "Pi as float:    " << piFloat << endl;
    cout << setprecision(64) << "Pi Intervall float : [" << iFloat.lower() << " | " << iFloat.upper() << endl;
    cout << setprecision(128) << "Pi Intervall double: [" << iDouble.lower() << " | " << iDouble.upper() << endl;
    cout << setprecision(128) << "test Intervall: [ " << test.lower() << " | " << test.upper() << " ]" << endl;

    if (test.lower() < 0.2) {
        cout << "Is lower" << endl;
    }

    if (test.upper() > 0.2) {
        cout << "Is higher" << endl;
    }


    return 0;
}