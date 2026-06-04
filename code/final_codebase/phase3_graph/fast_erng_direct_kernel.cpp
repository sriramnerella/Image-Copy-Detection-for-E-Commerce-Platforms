#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

extern "C" {

void direct_l2_search(
    const float* queries,
    int nq,
    const float* bank,
    int nb,
    int dim,
    float* out_d2,
    int* out_i
) {
    for (int q = 0; q < nq; ++q) {
        const float* qv = queries + static_cast<long long>(q) * dim;
        float best = std::numeric_limits<float>::infinity();
        int best_i = -1;
        for (int b = 0; b < nb; ++b) {
            const float* bv = bank + static_cast<long long>(b) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = qv[d] - bv[d];
                acc += diff * diff;
            }
            if (acc < best) {
                best = acc;
                best_i = b;
            }
        }
        out_d2[q] = best;
        out_i[q] = best_i;
    }
}

void erng_graph_search(
    const float* queries,
    int nq,
    const float* bank,
    int nb,
    int dim,
    const int* adj,
    int degree,
    int entry,
    int ef,
    float* out_d2,
    int* out_i,
    int* out_seen
) {
    std::vector<int> marks(nb, 0);
    std::vector<int> touched;
    touched.reserve(std::min(nb, ef * degree + 1));

    for (int q = 0; q < nq; ++q) {
        const int stamp = q + 1;
        touched.clear();
        const float* qv = queries + static_cast<long long>(q) * dim;

        auto dist2 = [&](int idx) {
            const float* bv = bank + static_cast<long long>(idx) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = qv[d] - bv[d];
                acc += diff * diff;
            }
            return acc;
        };

        using Node = std::pair<float, int>;
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;

        float d0 = dist2(entry);
        heap.push({d0, entry});
        marks[entry] = stamp;
        touched.push_back(entry);

        float best = d0;
        int best_i = entry;
        int visited = 0;

        while (!heap.empty() && visited < ef) {
            auto [cur_d, idx] = heap.top();
            heap.pop();
            ++visited;
            if (cur_d < best) {
                best = cur_d;
                best_i = idx;
            }
            const int* row = adj + static_cast<long long>(idx) * degree;
            for (int j = 0; j < degree; ++j) {
                int nb_i = row[j];
                if (nb_i < 0 || nb_i >= nb || marks[nb_i] == stamp) {
                    continue;
                }
                marks[nb_i] = stamp;
                touched.push_back(nb_i);
                heap.push({dist2(nb_i), nb_i});
            }
        }
        out_d2[q] = best;
        out_i[q] = best_i;
        out_seen[q] = static_cast<int>(touched.size());
    }
}

}
