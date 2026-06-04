#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
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

void centroid_branch_bound_search(
    const float* queries,
    int nq,
    const float* bank,
    int nb,
    int dim,
    const float* centroids,
    int nc,
    const float* radii,
    const int* cluster_indices,
    const int* cluster_offsets,
    int init_k,
    float* out_d2,
    int* out_i,
    int* out_seen
) {
    std::vector<float> cdist2(nc);
    std::vector<float> lower2(nc);
    std::vector<int> order(nc);
    std::vector<char> touched(nc);

    for (int q = 0; q < nq; ++q) {
        const float* qv = queries + static_cast<long long>(q) * dim;
        std::fill(touched.begin(), touched.end(), 0);

        for (int c = 0; c < nc; ++c) {
            const float* cv = centroids + static_cast<long long>(c) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = qv[d] - cv[d];
                acc += diff * diff;
            }
            cdist2[c] = acc;
            float cd = std::sqrt(acc);
            float lb = cd - radii[c];
            lower2[c] = lb > 0.0f ? lb * lb : 0.0f;
            order[c] = c;
        }

        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return cdist2[a] < cdist2[b];
        });

        float best = std::numeric_limits<float>::infinity();
        int best_i = -1;
        int seen = 0;
        int ik = init_k < nc ? init_k : nc;

        auto scan_cluster = [&](int c) {
            touched[c] = 1;
            int s = cluster_offsets[c];
            int e = cluster_offsets[c + 1];
            seen += (e - s);
            for (int p = s; p < e; ++p) {
                int b = cluster_indices[p];
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
        };

        for (int k = 0; k < ik; ++k) {
            scan_cluster(order[k]);
        }

        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return lower2[a] < lower2[b];
        });

        for (int pos = 0; pos < nc; ++pos) {
            int c = order[pos];
            if (lower2[c] >= best) {
                break;
            }
            if (touched[c]) {
                continue;
            }
            scan_cluster(c);
        }

        out_d2[q] = best;
        out_i[q] = best_i;
        out_seen[q] = seen;
    }
}

void centroid_threshold_decision_search(
    const float* queries,
    int nq,
    const float* bank,
    int nb,
    int dim,
    const float* centroids,
    int nc,
    const float* radii,
    const int* cluster_indices,
    const int* cluster_offsets,
    float threshold2,
    int init_k,
    float* out_d2,
    int* out_i,
    int* out_seen,
    int* out_duplicate
) {
    std::vector<float> cdist2(nc);
    std::vector<float> lower2(nc);
    std::vector<int> order(nc);
    std::vector<char> touched(nc);

    for (int q = 0; q < nq; ++q) {
        const float* qv = queries + static_cast<long long>(q) * dim;
        std::fill(touched.begin(), touched.end(), 0);

        for (int c = 0; c < nc; ++c) {
            const float* cv = centroids + static_cast<long long>(c) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = qv[d] - cv[d];
                acc += diff * diff;
            }
            cdist2[c] = acc;
            float cd = std::sqrt(acc);
            float lb = cd - radii[c];
            lower2[c] = lb > 0.0f ? lb * lb : 0.0f;
            order[c] = c;
        }

        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return lower2[a] < lower2[b];
        });

        float best = std::numeric_limits<float>::infinity();
        int best_i = -1;
        int seen = 0;
        int dup = 0;

        auto scan_cluster = [&](int c) {
            touched[c] = 1;
            int s = cluster_offsets[c];
            int e = cluster_offsets[c + 1];
            seen += (e - s);
            for (int p = s; p < e; ++p) {
                int b = cluster_indices[p];
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
                if (acc <= threshold2) {
                    dup = 1;
                    return;
                }
            }
        };

        int ik = init_k < nc ? init_k : nc;
        for (int k = 0; k < ik && !dup; ++k) {
            scan_cluster(order[k]);
        }

        for (int pos = 0; pos < nc && !dup; ++pos) {
            int c = order[pos];
            if (lower2[c] > threshold2) {
                break; // no remaining cluster can contain a duplicate
            }
            if (touched[c]) {
                continue;
            }
            scan_cluster(c);
        }

        out_d2[q] = best;
        out_i[q] = best_i;
        out_seen[q] = seen;
        out_duplicate[q] = dup;
    }
}

void centroid_threshold_decision_search_seeded(
    const float* queries,
    int nq,
    const float* bank,
    int nb,
    int dim,
    const float* centroids,
    int nc,
    const float* radii,
    const int* cluster_indices,
    const int* cluster_offsets,
    const int* seed_centroids,
    float threshold2,
    int init_k,
    float* out_d2,
    int* out_i,
    int* out_seen,
    int* out_duplicate
) {
    std::vector<float> cdist2(nc);
    std::vector<float> lower2(nc);
    std::vector<int> order(nc);
    std::vector<char> touched(nc);

    for (int q = 0; q < nq; ++q) {
        const float* qv = queries + static_cast<long long>(q) * dim;
        std::fill(touched.begin(), touched.end(), 0);

        for (int c = 0; c < nc; ++c) {
            const float* cv = centroids + static_cast<long long>(c) * dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = qv[d] - cv[d];
                acc += diff * diff;
            }
            cdist2[c] = acc;
            float cd = std::sqrt(acc);
            float lb = cd - radii[c];
            lower2[c] = lb > 0.0f ? lb * lb : 0.0f;
            order[c] = c;
        }

        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return lower2[a] < lower2[b];
        });

        float best = std::numeric_limits<float>::infinity();
        int best_i = -1;
        int seen = 0;
        int dup = 0;

        auto scan_cluster = [&](int c) {
            if (c < 0 || c >= nc || touched[c]) return;
            touched[c] = 1;
            int s = cluster_offsets[c];
            int e = cluster_offsets[c + 1];
            seen += (e - s);
            for (int p = s; p < e; ++p) {
                int b = cluster_indices[p];
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
                if (acc <= threshold2) {
                    dup = 1;
                    return;
                }
            }
        };

        scan_cluster(seed_centroids[q]);
        int ik = init_k < nc ? init_k : nc;
        for (int k = 0; k < ik && !dup; ++k) {
            scan_cluster(order[k]);
        }
        for (int pos = 0; pos < nc && !dup; ++pos) {
            int c = order[pos];
            if (lower2[c] > threshold2) {
                break;
            }
            scan_cluster(c);
        }

        out_d2[q] = best;
        out_i[q] = best_i;
        out_seen[q] = seen;
        out_duplicate[q] = dup;
    }
}

}
