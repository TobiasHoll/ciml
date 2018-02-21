#ifndef HOPFIELD_H
#define HOPFIELD_H

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

namespace sign
{
    constexpr int8_t NEGATIVE = -1;
    constexpr int8_t POSITIVE =  1;
}

enum class hopfield_learning_rule
{
    HEBBIAN,
    HEBBIAN_UNLEARNING,
    STORKEY_VALABREGUE,
    PSEUDO_INVERSE
};

enum class hopfield_policy
{
    ASYNCHRONOUS,
    SYNCHRONOUS,
};

class hopfield_network
{
public:
    using pattern_t  = std::vector<int8_t>;
    using patterns_t = std::vector<std::shared_ptr<pattern_t>>;

    /*! \brief Trains the network using the specified learning rule */
    void learn(hopfield_learning_rule rule, const patterns_t& patterns)
    {
        m_patterns = patterns;
        m_rule = rule;

        switch (m_rule)
        {
            case hopfield_learning_rule::HEBBIAN:
                learn_hebbian();
                break;
            default:
                throw std::logic_error("Invalid learning rule");
        }
    }

    /*! \brief Queries the network for a specific pattern */
    template <typename Pattern>
    size_t query(hopfield_policy policy, Pattern&& input, size_t iterations, size_t check_every, size_t random_flip = 0) const
    {
        pattern_t pattern = std::forward<Pattern>(input);

        switch (policy)
        {
            case hopfield_policy::ASYNCHRONOUS:
                return query_async(pattern, iterations, check_every, random_flip);
            case hopfield_policy::SYNCHRONOUS:
                return query_sync(pattern, iterations, check_every, random_flip);
            default:
                throw std::logic_error("Invalid query policy");
        }
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

    /*! \brief Queries the Hopfield network using asynchronous (random) updates */
    size_t query_async(pattern_t& pattern, size_t iterations, size_t check_every, size_t random_flip) const
    {
        std::random_device rd;
        std::mt19937_64 generator{rd()};

        std::uniform_int_distribution<size_t> index_distribution(0, m_nodes - 1);

        for (size_t iteration = 1; iteration <= iterations; ++iteration)
        {
            // Update m_nodes at random
            size_t updates_performed = 0;
            for (size_t update = 0; update < m_nodes; ++update)
                updates_performed += fire_async(pattern, index_distribution(generator));

            // Every check_every iterations and at the very end, check if we found an existing pattern
            if (iteration % check_every == 0 || iteration == iterations)
                for (size_t reference = 0; reference < m_patterns.size(); ++reference)
                    if (std::equal(pattern.begin(), pattern.end(), m_patterns[reference]->begin()))
                        return reference;

            // Introduce noise in a minimum
            if (updates_performed == 0 && random_flip > 0)
                for (size_t update = 0; update < random_flip; ++update)
                    flip(pattern[index_distribution(generator)]);
        }
        return std::numeric_limits<size_t>::max();
    }

    /*! \brief Queries the Hopfield network using synchronous updates */
    size_t query_sync(pattern_t& pattern, size_t iterations, size_t check_every, size_t random_flip) const
    {
        std::random_device rd;
        std::mt19937_64 generator{rd()};

        std::uniform_int_distribution<size_t> index_distribution(0, m_nodes - 1);

        for (size_t iteration = 1; iteration <= iterations; ++iteration)
        {
            // Update m_nodes at random
            size_t updates_performed = fire_sync(pattern);

            // Every check_every iterations and at the very end, check if we found an existing pattern
            if (iteration % check_every == 0 || iteration == iterations)
                for (size_t reference = 0; reference < m_patterns.size(); ++reference)
                    if (std::equal(pattern.begin(), pattern.end(), m_patterns[reference]->begin()))
                        return reference;

            // Introduce noise in a minimum
            if (updates_performed == 0 && random_flip > 0)
                for (size_t update = 0; update < random_flip; ++update)
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
        return value >= 0.0 ? sign::POSITIVE : sign::NEGATIVE;
    }

    size_t                 m_nodes;
    std::vector<double>    m_weights;
    hopfield_learning_rule m_rule;
    patterns_t             m_patterns;
};

#endif // HOPFIELD_H
