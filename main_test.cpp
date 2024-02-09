#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <tracy/Tracy.hpp>


void performSleep(int sleep_duration) {
    ZoneScoped;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration));
}

void worker(int id) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(1000, 5000);

    while (true) {
        int sleep_duration = distr(gen); // Random duration between 1 and 5 seconds
        performSleep(sleep_duration);
        std::cout << "Thread " << id << " slept for " << sleep_duration << " milliseconds." << std::endl;
    }
}

int main() {
    std::thread thread1(worker, 1);
    std::thread thread2(worker, 2);

    thread1.join();
    thread2.join();

    return 0;
}
