#ifndef HOPFIELD_H
#define HOPFIELD_H

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <variant>
#include <vector>

/* Learning rules */
enum class HopfieldLearningRule
{
    HEBBIAN,
    STORKEY_VALABREGUE,
    PSEUDO_INVERSE
};

/* Query policies */
namespace HopfieldPolicy
{
    struct Synchronous
    {
        size_t iterations;
        size_t check_every;
        size_t random_flip;
    };
    struct Asynchronous
    {
        size_t iterations;
        size_t updates_per_iteration;
        size_t check_every;
        size_t random_flip;
    };
}

/*! \brief A hopfield network */
class HopfieldNetwork
{
public:
    using pattern_t  = std::vector<int8_t>;
    using patterns_t = std::vector<std::shared_ptr<pattern_t>>;

    /*! \brief Trains the network using the specified learning rule */
    void learn(HopfieldLearningRule rule, const patterns_t& patterns)
    {
        m_patterns = patterns;

        switch (rule)
        {
            case HopfieldLearningRule::HEBBIAN:
                return learn_hebbian();
            case HopfieldLearningRule::STORKEY_VALABREGUE:
                return learn_storkey_valabregue();
            case HopfieldLearningRule::PSEUDO_INVERSE:
                return learn_pseudo_inverse();
            default:
                throw std::logic_error("Invalid learning rule");
        }
    }

    /*! \brief Queries the network for a specific pattern */
    template <typename Policy, typename Pattern>
    size_t query(Policy&& policy, Pattern&& input) const
    {
        pattern_t pattern = std::forward<Pattern>(input);
        return query_impl(policy, pattern);
    }

private:
    /*! \brief Trains the Hopfield network using Hebbian learning */
    void learn_hebbian()
    {
        if (m_patterns.size() <= 0)
            throw std::logic_error("No patterns specified");

        m_nodes = m_patterns[0]->size();
        m_weights = std::vector<double>(m_nodes * m_nodes);

        for (size_t pattern = 0; pattern < m_patterns.size(); ++pattern)
        {
            if (m_patterns[pattern]->size() != m_nodes)
                throw std::logic_error("Invalid pattern size");

            for (size_t i = 0; i < m_nodes; ++i)
            {
                int8_t ei = m_patterns[pattern]->operator[](i);
                for (size_t j = 0; j < i; ++j)
                {
                    int8_t ej = m_patterns[pattern]->operator[](j);
                    double modifier = (double) (ei * ej) / (double) m_patterns.size();
                    m_weights[i * m_nodes + j] += modifier;
                    m_weights[j * m_nodes + i] += modifier;
                }
            }
        }
    }

    /*! \brief Trains the Hopfield network using Storkey-Valabregue learning */
    void learn_storkey_valabregue()
    {
        if (m_patterns.size() <= 0)
            throw std::logic_error("No patterns specified");

        m_nodes = m_patterns[0]->size();
        m_weights = std::vector<double>(m_nodes * m_nodes);

        for (size_t pattern = 0; pattern < m_patterns.size(); ++pattern)
        {
            if (m_patterns[pattern]->size() != m_nodes)
                throw std::logic_error("Invalid pattern size");

            std::vector<double> next_weights(m_nodes * m_nodes);
            for (size_t i = 0; i < m_nodes; ++i)
            {
                int8_t ei = m_patterns[pattern]->operator[](i);
                for (size_t j = 0; j < m_nodes; ++j)
                {
                    int8_t ej = m_patterns[pattern]->operator[](j);

                    double hij = 0.0;
                    double hji = 0.0;
                    for (size_t k = 0; k < m_nodes; ++k)
                    {
                        if (k != i && k != j)
                        {
                            int8_t ek = m_patterns[pattern]->operator[](k);
                            hij += m_weights[i * m_nodes + k] * ek;
                            hji += m_weights[j * m_nodes + k] * ek;
                        }
                    }

                    double modifier = (double) (ei * ej - ei * hij - ej * hji) / (double) m_nodes;
                    next_weights[i * m_nodes + j] = m_weights[i * m_nodes + j] + modifier;
                }
            }
            m_weights = std::move(next_weights);
        }
    }

    /*! \brief Trains the Hopfield network using pseudoinverse learning */
    void learn_pseudo_inverse()
    {
        throw std::logic_error("Not implemented");
    }

    /*! \brief Queries the Hopfield network using asynchronous (random) updates */
    size_t query_impl(const HopfieldPolicy::Asynchronous& policy, pattern_t& pattern) const
    {
        std::random_device rd;
        std::mt19937_64 generator{rd()};

        std::uniform_int_distribution<size_t> index_distribution(0, m_nodes - 1);

        for (size_t iteration = 1; iteration <= policy.iterations; ++iteration)
        {
            // Update nodes at random
            size_t updates_performed = 0;
            for (size_t update = 0; update < policy.updates_per_iteration; ++update)
                updates_performed += fire_async(pattern, index_distribution(generator));

            // Every check_every iterations and at the very end, check if we found an existing pattern
            if (iteration % policy.check_every == 0 || iteration == policy.iterations)
                for (size_t reference = 0; reference < m_patterns.size(); ++reference)
                    if (std::equal(pattern.begin(), pattern.end(), m_patterns[reference]->begin()))
                        return reference;

            // Introduce noise in a minimum
            if (updates_performed == 0 && policy.random_flip > 0)
                for (size_t update = 0; update < policy.random_flip; ++update)
                    flip(pattern[index_distribution(generator)]);
        }
        return std::numeric_limits<size_t>::max();
    }

    /*! \brief Queries the Hopfield network using synchronous updates */
    size_t query_impl(const HopfieldPolicy::Synchronous& policy, pattern_t& pattern) const
    {
        std::random_device rd;
        std::mt19937_64 generator{rd()};

        std::uniform_int_distribution<size_t> index_distribution(0, m_nodes - 1);

        for (size_t iteration = 1; iteration <= policy.iterations; ++iteration)
        {
            // Update m_nodes at random
            size_t updates_performed = fire_sync(pattern);

            // Every check_every iterations and at the very end, check if we found an existing pattern
            if (iteration % policy.check_every == 0 || iteration == policy.iterations)
                for (size_t reference = 0; reference < m_patterns.size(); ++reference)
                    if (std::equal(pattern.begin(), pattern.end(), m_patterns[reference]->begin()))
                        return reference;

            // Introduce noise in a minimum
            if (updates_performed == 0 && policy.random_flip > 0)
                for (size_t update = 0; update < policy.random_flip; ++update)
                    flip(pattern[index_distribution(generator)]);
        }
        return std::numeric_limits<size_t>::max();
    }

    /*! \brief Flips a bit */
    static inline void flip(int8_t& sgn)
    {
        sgn = (int8_t) (((int) sgn) * (-1));
    }

    /*! \brief Fires the specified node asynchronously */
    size_t fire_async(pattern_t& pattern, size_t index) const
    {
        int8_t new_value = compute(pattern, index);
        if (new_value != pattern[index])
        {
            pattern[index] = new_value;
            return 1;
        }
        else return 0;
    }

    /* \brief Fires all nodes synchronously */
    size_t fire_sync(pattern_t& pattern) const
    {
        pattern_t replacements(m_nodes);
        for (size_t node = 0; node < m_nodes; ++node)
            replacements[node] = compute(pattern, node);

        size_t updates_performed = 0;
        for (size_t node = 0; node < m_nodes; ++node)
        {
            if (replacements[node] != pattern[node])
            {
                ++updates_performed;
                pattern[node] = replacements[node];
            }
        }
        return updates_performed;
    }

    /* \brief Computes the result of a single node */
    int8_t compute(pattern_t& pattern, size_t index) const
    {
        double value = 0.0;
        for (size_t node = 0; node < m_nodes; ++node)
            value += pattern[node] * m_weights[index * m_nodes + node];
        return value >= 0.0 ? POSITIVE : NEGATIVE;
    }

    constexpr static int8_t POSITIVE = 1;
    constexpr static int8_t NEGATIVE = -1;

    size_t               m_nodes;
    std::vector<double>  m_weights;
    patterns_t           m_patterns;
};

#endif // HOPFIELD_H
