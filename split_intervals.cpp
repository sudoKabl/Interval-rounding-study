#include <iostream>
#include <vector>
#include <functional>
#include <thread>
#include <future>
#include <iomanip>
#include <chrono>

#include <boost/numeric/interval.hpp>
#include <boost/numeric/interval/io.hpp>
#include <boost/numeric/interval/ext/integer.hpp>

#include <boost/numeric/interval/limits.hpp>


/*
Notizen:
Doc von allen Functionsdefs:
https://www.boost.org/doc/libs/1_85_0/libs/numeric/interval/doc/interval.htm

Headers mit erklärungen:
https://www.boost.org/doc/libs/1_84_0/libs/numeric/interval/doc/includes.htm

*/

using namespace boost::numeric;
using namespace std;

typedef interval<double> Interval;




// Primary functions
Interval withLoop(Interval x, string w, function<void(const Interval&, Interval*)> f);
Interval withRecursion(Interval x, double w, function<void(const Interval&, Interval*)> f);


// Helper functions
vector<Interval> split(Interval& i, double at);
Interval cUnion(Interval& a, Interval& b);

Interval computeInterval(

    const function<Interval(const Interval&)>& f,
    const Interval& x) {

    return f(x);
}

// ------------------------------------------------------------------------------------------------
// ----- Main function
// ------------------------------------------------------------------------------------------------

int main() {

    // Example function f: x^2 + x

    auto f = [](const Interval& x, Interval* r) -> void {
        Interval result = pow(x, 2) + x;
        r->assign(result.lower(), result.upper());
        };


    // Example function f: 2x^3 + 3x^2 - 2x + 1
    auto h = [](const Interval& x, Interval* r) -> void {

        Interval result_a = 2 * pow(x, 3);
        
        Interval result_b = 3 * pow(x, 2);

        Interval result_c = 2 * x;

        Interval result = result_a + result_b - result_c + Interval(1, 1);

        r->assign(result.lower(), result.upper());


        return;
    };

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // ----- Auswertung von h
    Interval xf = Interval(-1, 1);
    const int len = 11;
    double avr_loop[16][len];
    double avr_val[len];

    string w_string[] = { "2", "1", "0.5", "0.2", "0.1", "0.05", "0.01", "0.008", "0.005", "0.001", "0.0005"};
    
    cout << "===========" << endl;
    cout << "Loop results: " << endl;

    // Für mehrere Durchläufe kann hier j erhöht werden
    int num_runs = 1;

    for (int j = 0; j < num_runs; j++) {
        for (int i = 0; i < len; i++) {
            auto start = chrono::high_resolution_clock::now();

            Interval loopResult = withLoop(xf, w_string[i], h);

            auto stop = chrono::high_resolution_clock::now();
            //auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

            duration<double, std::milli> ms_double = stop - start;
            avr_loop[j][i] = ms_double.count();

            cout << setprecision(64) << "Result for " << w_string[i] << " : " << loopResult << endl;
            //cout << "Execution time: " << ms_double.count() << " ms" << endl << "--------" << endl;
        }
    }

    for (int i = 0; i < len; i++) {
        double sum = 0;
        for (int j = 0; j < num_runs; j++) {
            sum += avr_loop[j][i];
        }
        avr_val[i] = sum / num_runs;
        cout << setprecision(64) << "Average time for " << w_string[i] << ": " << avr_val[i] << endl;
    }
    

    
    double w_double[] = { 2, 1, 0.5, 0.1, 0.01, 0.005 };

    cout << "===========" << endl;
    cout << "Recursive results: " << endl;
    
    for (int i = 0; i < len; i++) {
        auto start = chrono::high_resolution_clock::now();

        Interval recursiveResult = withRecursion(xf, w_double[i], h);

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

        cout << "Result for " << w_double[i] << " : " << recursiveResult << endl;
        cout << "Execution time: " << duration.count() << " ms" << endl << "--------" << endl;
    }

    return 0;
}

// ------------------------------------------------------------------------------------------------
// ----- Primary functions
// ------------------------------------------------------------------------------------------------


Interval withLoop(Interval x, string w, function<void(const Interval&, Interval*)> f) {

    // First convert double to int, to combat rounding floating point errors:

    int scale = 1;
    double w_val = stod(w);

    // Only neccessary when there actually are decimal places
    if ((int)w_val != w_val) {
        int lenAfterDot = w.length();
        int i = 0;

        // Check length of decimal places
        for (lenAfterDot; lenAfterDot > 0; lenAfterDot--) {

            if (w[i] == '.') {
                break;
            }
            i++;
        }

        // Scale value so we only have whole numbers
        scale = pow(10, lenAfterDot-1);
        w_val *= scale;
    }

    //cout << w_val << " - " << scale << endl;
    
    double scaled_width = width(x) * scale;

    // Calculate how many intervals fit
    double needed_intervals_calc = scaled_width / w_val;
    int needed_intervals = (int)needed_intervals_calc + ((int)needed_intervals_calc != needed_intervals_calc ? 1 : 0);

    //cout << "Calc: " << needed_intervals_calc << " -> " << needed_intervals << endl;

    double target_interval_size = scaled_width / needed_intervals;

    //cout << target_interval_size << endl;

    // Build vector of all sub-intervals
    vector<Interval> split_results = { Interval(x.lower() * scale, x.lower() * scale + target_interval_size) };

    for (int i = 1; i < needed_intervals; i++) {
        split_results.push_back(Interval(split_results[i - 1].upper(), split_results[i - 1].upper() + target_interval_size));
    }

    // Multi-Threaded application of f(x)

    vector<thread*> open_threads;
    vector<Interval> thread_results;

    // Allocate result vector thread_results
    for (size_t i = 0; i < split_results.size(); i++) {
        thread_results.push_back(Interval(0, 0));
    }
    
    // Call indivdual threads, keep track in open_threads, revert previous scaling
    for (size_t i = 0; i < split_results.size(); i++) {

        open_threads.push_back(new thread(f, Interval(split_results[i].lower() / scale, split_results[i].upper() / scale), &thread_results[i]));
    }

    // Wait for all threads to finish
    for (size_t i = 0; i < split_results.size(); i++) {
        open_threads[i]->join();
        //cout << "Result of " << i << ": " << thread_results[i] << endl;
    }

    Interval final_result = thread_results.back();
    thread_results.pop_back();



    while (!thread_results.empty()) {
        //cout << thread_results.back() << " + " << final_result << " = ";
        final_result = cUnion(final_result, thread_results.back());
        thread_results.pop_back();
        //cout << final_result << endl;
    }
    
    //cout << "Final resulting Interval from loop: " << final_result << endl;


    return final_result;
}

Interval splitOrUnion(Interval a, double minW, function<void(const Interval&, Interval*)> f) {
    double currentWidth = a.upper() - a.lower();
    if (currentWidth > minW) {
        vector<Interval> splits = split(a, a .lower() + currentWidth / 2);

        auto futureL = async(launch::async, splitOrUnion, splits[0], minW, f);
        auto futureU = async(launch::async, splitOrUnion, splits[1], minW, f);

        Interval l = futureL.get();
        Interval u = futureU.get();

        return cUnion(l, u);
    }
    else {
        Interval result;
        f(a, &result);
        return result;
    }

}

Interval withRecursion(Interval x, double w, function<void(const Interval&, Interval*)> f) {
    
    Interval final_result = splitOrUnion(x, w, f);
    //cout << "Final resulting Interval from recursion: " << final_result << endl;

    return final_result;
}



// ------------------------------------------------------------------------------------------------
// ----- Helper functions
// ------------------------------------------------------------------------------------------------


vector<Interval> split(Interval& i, double at) {

    if (!subset(Interval(at, at), i)) {
        cout << "ERROR: " << at << " is not inside of " << i << endl;
        exit(1);
    }

    vector<Interval> result = { Interval(i.lower(), at), Interval(at, i.upper()) };

    return result;
}

Interval cUnion(Interval& a, Interval& b) {

    double lower = (a.lower() < b.lower() ? a.lower() : b.lower());
    double upper = (a.upper() > b.upper() ? a.upper() : b.upper());

    return Interval(lower, upper);
}