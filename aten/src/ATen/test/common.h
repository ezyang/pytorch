#include <chrono>

void report_header() {
  std::cout << std::setw(55) << "name"
            << std::setw(25) << "samples/sec (avg)"
            << std::setw(25) << "ns (min)"
            << std::setw(25) << "stdev"
            << std::endl;
}

void report(const std::string& name,
            double duration_avg,
            double duration_min,
            double stdev) {
  std::cout << std::setw(55) << std::fixed << std::setprecision(2)
            << name
            << std::setw(25) << 1 / duration_avg
            << std::setw(25) << duration_min * 1e9
            << std::setw(25) << stdev * 1e9
            << std::endl;
}

class Timer {
public:
  Timer(long custom_iter_count) : custom_iter_count_(custom_iter_count) {}

  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void stop(long iter_count) {
    auto now = std::chrono::high_resolution_clock::now();
    durations_.push_back(
        std::chrono::duration_cast<std::chrono::duration<double>>(
            now - start_).count() / iter_count / custom_iter_count_);
  }

  double min() {
    return *std::min_element(durations_.begin(), durations_.end());
  }

  double mean() {
    double sum = std::accumulate(durations_.begin(), durations_.end(), 0.0);
    return sum / durations_.size();
  }

  double stdev() {
    double m = mean();
    double sum = 0.0;
    std::for_each(std::begin(durations_), std::end(durations_), [&](const double& v) {
      sum += (v - m) * (v - m);
    });
    return sqrt(sum / (durations_.size() - 1));
  }

private:
  long custom_iter_count_;
  std::vector<double> durations_;
  std::chrono::high_resolution_clock::time_point start_;
};

void benchmark(
    const std::string& name,
    long iter_count,
    long custom_iter_count,
    const std::function<void()>& fun) {
  for (int i = 0; i < iter_count / 2; ++i) {  /* warm up */
    fun();
  }
  Timer t(custom_iter_count);
  for (int i = 0; i < 10; ++i) {
    t.start();
    for (int i = 0; i < iter_count / 10; ++i) {
      fun();
    }
    t.stop(iter_count / 10);
  }
  report(name, t.mean(), t.min(), t.stdev());
}

void benchmark(
    const std::string& name,
    long iter_count,
    const std::function<void()>& fun) {
  benchmark(name, iter_count, 1, fun);
}
