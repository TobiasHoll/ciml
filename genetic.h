#ifndef CIML_GENETIC_H
#define CIML_GENETIC_H

#ifndef CIML_RANDOM
    #define CIML_RANDOM std::mt19937_64
    #define BOOST_ASYNCHRONOUS_USE_STD_RANDOM
#endif

#include <set>
#include <type_traits>
#include <vector>

#include <boost/asynchronous/servant_proxy.hpp>
#include <boost/asynchronous/trackable_servant.hpp>
#include <boost/asynchronous/algorithm/parallel_for.hpp>
#include <boost/asynchronous/algorithm/parallel_generate.hpp>
#include <boost/asynchronous/algorithm/parallel_reduce.hpp>
#include <boost/asynchronous/algorithm/parallel_sort.hpp>
#include <boost/asynchronous/algorithm/parallel_transform.hpp>
#include <boost/asynchronous/algorithm/then.hpp>
#include <boost/asynchronous/container/algorithms.hpp>
#include <boost/asynchronous/container/vector.hpp>
#include <boost/asynchronous/helpers/random_provider.hpp>


namespace ciml::genetic
{
    
using Random = boost::asynchronous::random_provider<CIML_RANDOM>;

template <typename Chromosome, typename Fitness>
class TruncationSelector
{
public:
    constexpr static bool require_total_fitness = false; 
    using FitnessValue = std::invoke_result_t<Fitness, Chromosome>;
    using RankedChromosome = std::pair<FitnessValue, Chromosome>;
    
    TruncationSelector(double ratio) : m_ratio(ratio) {}
    
    template <typename Container>
    std::pair<RankedChromosome, RankedChromosome> operator()(const Container& container, FitnessValue) const
    {
        std::uniform_int_distribution<size_t> index_selector(0, static_cast<size_t>(m_ratio * (container.size() - 1)));
        return { container[Random::generate(index_selector)], container[Random::generate(index_selector)] };
    }
    
private:
    double m_ratio;
};

template <typename Chromosome, typename Fitness>
class FitnessSelector
{
public:
    constexpr static bool require_total_fitness = true;
    using FitnessValue = std::invoke_result_t<Fitness, Chromosome>;
    using RankedChromosome = std::pair<FitnessValue, Chromosome>;
    
    // "Roulette-wheel selection via stochastic acceptance." Lipowski, A.; Lipowska, D. (2011), https://arxiv.org/pdf/1109.3627.pdf
    template <typename Container>
    static size_t random_index(const Container& container, double total_fitness)
    {
        std::uniform_int_distribution<size_t> index_selector(0, container.size() - 1);
        std::uniform_real_distribution<double> probability(0, 1);
        size_t index;
        do { index = Random::generate(index_selector); }
        while (Random::generate(probability) > (static_cast<double>(container[index].first) / total_fitness));
        return index;
    }
    
    template <typename Container>
    std::pair<RankedChromosome, RankedChromosome> operator()(const Container& container, FitnessValue total_fitness) const
    {
        size_t a = random_index(container, static_cast<double>(total_fitness));
        size_t b = random_index(container, static_cast<double>(total_fitness));
        return { container[a], container[b] };
    }
};

template <typename Chromosome, typename Fitness>
class RankSelector
{
public:
    constexpr static bool require_total_fitness = false;
    using FitnessValue = std::invoke_result_t<Fitness, Chromosome>;
    using RankedChromosome = std::pair<FitnessValue, Chromosome>;
    
    // Similar to the fitness-based selector, but based on rank instead.
    static size_t random_index(size_t size)
    {
        // The item at index size - x has x "tokens" in the pool that can be pulled.
        // To translate the "index" of such a token back to an actual index, we need to inverse the triangular numbers
        // i(n) = floor(sqrt(2 * n) + 0.5).
        std::uniform_int_distribution<size_t> distribution(0, size * (size - 1) - 1);
        size_t token_index = Random::generate(distribution);
        return static_cast<size_t>(std::floor(std::sqrt(2.0 * token_index) + 0.5));
    }
    
    template <typename Container>
    std::pair<RankedChromosome, RankedChromosome> operator()(const Container& container, FitnessValue) const
    {
        size_t a = random_index(container.size());
        size_t b = random_index(container.size());
        return { container[a], container[b] };
    }
};

template <typename Chromosome, typename Fitness>
class TournamentSelector
{
public:
    constexpr static bool require_total_fitness = false;
    using FitnessValue = std::invoke_result_t<Fitness, Chromosome>;
    using RankedChromosome = std::pair<FitnessValue, Chromosome>;
    
    TournamentSelector(size_t size, double probability) : m_tournament_size(size), m_probability(probability) {}
    
    // Similar to the fitness-based selector, but based on rank instead.
    size_t random_index(size_t size)
    {
        // Generate k (m_tournament_size) random indices, but keep them in order.
        std::set<size_t> indices;
        std::uniform_int_distribution<size_t> index_selector(0, size - 1);
        while (indices.size() < std::min(m_tournament_size, size))
            indices.insert(Random::generate(index_selector));
        
        // Select the first element with probability p (m_probability). If it is not selected, proceed for the next item until no more items are left.
        std::uniform_real_distribution<double> probability(0, 1);
        size_t index;
        for (const size_t& maybe_index : indices)
        {
            index = maybe_index;
            if (Random::generate(probability) < m_probability)
                break;
            // If we run out of items, index contains the last item, so we are fine.
        }
        return index;
    }
    
    template <typename Container>
    std::pair<RankedChromosome, RankedChromosome> operator()(const Container& container, FitnessValue) const
    {
        size_t a = random_index(container.size());
        size_t b = random_index(container.size());
        return { container[a], container[b] };
    }
    
private:
    size_t m_tournament_size;
    double m_probability;
};


template <
    typename Chromosome,
    typename Generator,
    typename Fitness,
    typename Selector,
    typename Crossover,
    typename Mutator,
    typename Job = BOOST_ASYNCHRONOUS_DEFAULT_JOB
>
class EvolutionServant : public boost::asynchronous::trackable_servant<Job, Job>
{
    constexpr static long MAKE_RANGE_CUTOFF               = 2000;
    constexpr static long GENERATOR_CUTOFF                =  500;
    constexpr static long FITNESS_CUTOFF                  =   10;
    constexpr static long SORT_CUTOFF                     = 2000;
    constexpr static long ELITISM_CUTOFF                  = 2000;
    constexpr static long SELECTION_AND_CROSSOVER_CUTOFF  =  500;
    constexpr static long MUTATION_CUTOFF                 =  500;
    
    constexpr static size_t CHILDREN_PER_SELECTED_PAIR = Crossover::CHILDREN;
    
#ifdef CIML_USE_ASYNCHRONOUS_VECTOR
    template <typename T> using Vector = boost::asynchronous::vector<T>;
#else
    template <typename T> using Vector = std::vector<T>;
#endif
    
    using Population = Vector<Chromosome>;
    using FitnessValue = std::invoke_result_t<Fitness, Chromosome>;
    using RankedChromosome = std::pair<FitnessValue, Chromosome>;
    using OrderedPopulation = Vector<RankedChromosome>;
    
    struct RankedChromosomeWrapper
    {
        RankedChromosomeWrapper() : value(0) {}
        RankedChromosomeWrapper(const RankedChromosome& rc) : value(rc.first) {}
        RankedChromosomeWrapper(const FitnessValue& fv) : value(fv) {}
        
        FitnessValue value;
    };

public:
    constexpr static size_t PRIO = 0;

    EvolutionServant(boost::asynchronous::any_weak_scheduler<Job> scheduler,
                     boost::asynchronous::any_shared_scheduler_proxy<Job> worker,
                     Generator generator,
                     Fitness   fitness,
                     Selector  selector,
                     Crossover crossover,
                     Mutator   mutator,
                     double    elitism,
                     double    replacement)
        : boost::asynchronous::trackable_servant<Job, Job>(scheduler, worker)
        , m_generator(std::move(generator))
        , m_fitness(std::move(fitness))
        , m_selector(std::move(selector))
        , m_crossover(std::move(crossover))
        , m_mutator(std::move(mutator))
        , m_elitism(elitism)
        , m_replacement(replacement)
    {}

    template <typename Callback>
    void generate(size_t count, Callback callback)
    {
        this->post_callback(
            [generator = m_generator, count]() mutable
            {
                return boost::asynchronous::parallel_generate(
                    Population(count),
                    std::move(generator),
                    GENERATOR_CUTOFF,
                    "EvolutionServant::EvolutionServant: parallel_generate",
                    PRIO
                );
            },
            [callback = std::move(callback)](boost::asynchronous::expected<Population> expected) mutable
            {
                callback(std::move(expected.get()));
            },
            "EvolutionServant::EvolutionServant: post_callback",
            PRIO,
            PRIO
        );
    }

    template <typename Callback>
    void cycle(Population population, Callback callback)
    {
        this->post_callback(
            [population = std::move(population), generator = m_generator, fitness = m_fitness, selector = m_selector, crossover = m_crossover, mutator = m_mutator, elitism = m_elitism, replacement = m_replacement]
            {
                return EvolutionServant::cycle(std::move(population), std::move(generator), std::move(fitness), std::move(selector), std::move(crossover), std::move(mutator), elitism, replacement);
            },
            [callback = std::move(callback)](boost::asynchronous::expected<Population> expected) mutable
            {
                callback(std::move(expected.get()));
            },
            "EvolutionServant::cycle: post_callback",
            PRIO,
            PRIO
        );
    }
    
private:
    static auto cycle(Population population, Generator generator, Fitness fitness, Selector selector, Crossover crossover, Mutator mutator, double elitism, double replacement)
    {
        auto population_ptr = std::make_shared<Population>(std::move(population));
        
        // How many pairs do we need to select later
        size_t elitism_n = static_cast<size_t>(elitism * population_ptr->size());
        size_t replacement_n = static_cast<size_t>(replacement * population_ptr->size());
        size_t select_n = static_cast<size_t>(std::ceil(static_cast<double>(population_ptr->size() - elitism_n - replacement_n) / CHILDREN_PER_SELECTED_PAIR));
        size_t next_generation_size = elitism_n + replacement_n + select_n * CHILDREN_PER_SELECTED_PAIR;

        // First, assign fitness to everything and sort
        std::shared_ptr<OrderedPopulation> ordered_ptr = std::make_shared<OrderedPopulation>(population_ptr->size());
        auto ordered_population_cont = boost::asynchronous::then(
            boost::asynchronous::parallel_transform(
                population_ptr->begin(),
                population_ptr->end(),
                ordered_ptr->begin(),
                [fitness = std::move(fitness)](const Chromosome& chromosome) mutable { return RankedChromosome(fitness(chromosome), chromosome); },
                FITNESS_CUTOFF,
                "EvolutionServant::cycle: compute fitness",
                PRIO
            ),
            [population_ptr, ordered_ptr](boost::asynchronous::expected<typename OrderedPopulation::iterator> expected) mutable
            {
                expected.get();
                return boost::asynchronous::parallel_sort(
                    ordered_ptr->begin(),
                    ordered_ptr->end(),
                    [](const RankedChromosome& a, const RankedChromosome& b) { return a.first > b.first; },
                    SORT_CUTOFF,
                    "EvolutionServant::cycle: rank chromosomes",
                    PRIO
                );
            },
            "EvolutionServant::cycle: then 1 (ranking)"
        );
        
        // Compute generation fitness
        auto total_fitness_ptr = std::make_shared<FitnessValue>(static_cast<FitnessValue>(0));
        if constexpr (Selector::require_total_fitness)
        {
            ordered_population_cont = boost::asynchronous::then(
                ordered_population_cont,
                [ordered_ptr, total_fitness_ptr](auto expected)
                {
                    expected.get();
                    // Single-threaded because parallel_reduce is being annoying again
                    for (const auto& ranked : *ordered_ptr)
                        *total_fitness_ptr += ranked.first;
                },
                "EvolutionServant::cycle: then 2 (total fitness computation)"
            );
        }

        // Make space for the new generation
        std::shared_ptr<Population> next_generation_ptr = std::make_shared<Population>(next_generation_size);
        
        // Copy over the top elitism_n chromosomes
        auto elitism_cont = boost::asynchronous::then(
            ordered_population_cont,
            [ordered_ptr, next_generation_ptr, elitism_n](auto expected) mutable
            {
                expected.get();
                return boost::asynchronous::parallel_transform(
                    ordered_ptr->begin(),
                    ordered_ptr->begin() + elitism_n,
                    next_generation_ptr->begin(),
                    [](const RankedChromosome& ranked) { return ranked.second; },
                    ELITISM_CUTOFF,
                    "EvolutionServant::cycle: elitism",
                    PRIO
                );
            },
            "EvolutionServant::cycle: then 3 (elitism)"
        );
        
        // Then, call the user-specified selector and crossover to build the new generation
        auto selection_cont = boost::asynchronous::then(
            elitism_cont,
            [ordered_ptr, next_generation_ptr, total_fitness_ptr, elitism_n, select_n, selector = std::move(selector), crossover = std::move(crossover)](auto expected) mutable
            {
                expected.get();
                return boost::asynchronous::parallel_for(
                    static_cast<size_t>(0),
                    select_n,
                    [ordered_ptr, next_generation_ptr, total_fitness_ptr, elitism_n, select_n, selector = std::move(selector), crossover = std::move(crossover)](const size_t& index) mutable
                    {
                        auto [a, b] = selector(*ordered_ptr, *total_fitness_ptr);
                        auto children = crossover(std::move(a.second), std::move(b.second));
                        std::move(children.begin(), children.end(), next_generation_ptr->begin() + elitism_n + index * CHILDREN_PER_SELECTED_PAIR);
                    },
                    SELECTION_AND_CROSSOVER_CUTOFF,
                    "EvolutionServant::cycle: selection and crossover"
                );
            },
            "EvolutionServant::cycle: then 4 (selection and crossover)"
        );
        
        // Add replacement_n randomly created chromosomes
        auto replacement_cont = boost::asynchronous::then(
            selection_cont,
            [next_generation_ptr, elitism_n, select_n, generator = std::move(generator)](auto expected) mutable
            {
                expected.get();
                boost::asynchronous::parallel_generate(
                    next_generation_ptr->begin() + elitism_n + select_n * CHILDREN_PER_SELECTED_PAIR,
                    next_generation_ptr->end(),
                    std::move(generator),
                    GENERATOR_CUTOFF,
                    "EvolutionServant::cycle: generate replacements",
                    PRIO
                );
            },
            "EvolutionServant::cycle: then 5 (replacement)"
        );

        // Finally, use the mutator to modify the new chromosomes (but not those preserved by elitism, nor those newly created).
        auto mutation_cont = boost::asynchronous::then(
            replacement_cont,
            [next_generation_ptr, elitism_n, select_n, mutator = std::move(mutator)](auto expected) mutable
            {
                expected.get();
                return boost::asynchronous::parallel_for(
                    next_generation_ptr->begin() + elitism_n,
                    next_generation_ptr->begin() + elitism_n + select_n * CHILDREN_PER_SELECTED_PAIR,
                    std::move(mutator),
                    MUTATION_CUTOFF,
                    "EvolutionServant::cycle: mutation"
                );
            },
            "EvolutionServant::cycle: then 6 (mutation)"
        );
        
        // Unwrap the result
        return boost::asynchronous::then(
            mutation_cont,
            [next_generation_ptr](auto expected) mutable
            {
                expected.get();
                return *next_generation_ptr;
            },
            "EvolutionServant::cycle: then 7 (unwrap)"
        );
    }

    Generator m_generator;
    Fitness   m_fitness;
    Selector  m_selector;
    Crossover m_crossover;
    Mutator   m_mutator;
    double    m_elitism;
    double    m_replacement;
};

template <
    typename Chromosome,
    typename Generator,
    typename Fitness,
    typename Selector,
    typename Crossover,
    typename Mutator,
    typename Job = BOOST_ASYNCHRONOUS_DEFAULT_JOB
>
class EvolutionProxy
    : public boost::asynchronous::servant_proxy<
          EvolutionProxy<Chromosome, Generator, Fitness, Selector, Crossover, Mutator, Job>,
          EvolutionServant<Chromosome, Generator, Fitness, Selector, Crossover, Mutator, Job>
      >
{
public:
    using proxy_type   = EvolutionProxy<Chromosome, Generator, Fitness, Selector, Crossover, Mutator, Job>;
    using servant_type = EvolutionServant<Chromosome, Generator, Fitness, Selector, Crossover, Mutator, Job>;
    
    using callable_type = typename proxy_type::callable_type;
    
    template <typename Scheduler, typename Worker>
    EvolutionProxy(Scheduler scheduler,
                   Worker worker,
                   Generator generator   = Generator(),
                   Fitness   fitness     = Fitness(),
                   Selector  selector    = Selector(),
                   Crossover crossover   = Crossover(),
                   Mutator   mutator     = Mutator(),
                   double    elitism     = 0.01,
                   double    replacement = 0.1)
        : boost::asynchronous::servant_proxy<proxy_type, servant_type>(
              std::move(scheduler),
              std::move(worker),
              std::move(generator),
              std::move(fitness),
              std::move(selector),
              std::move(crossover),
              std::move(mutator),
              std::move(elitism),
              std::move(replacement)
          )
    {}

    BOOST_ASYNC_SERVANT_POST_CTOR_LOG("EvolutionProxy: constructor", servant_type::PRIO)
    BOOST_ASYNC_SERVANT_POST_DTOR_LOG("EvolutionProxy: destructor",  servant_type::PRIO)
    
    BOOST_ASYNC_POST_MEMBER_LOG(generate, "EvolutionProxy::generate", servant_type::PRIO)
    BOOST_ASYNC_POST_MEMBER_LOG(cycle,    "EvolutionProxy::cycle",    servant_type::PRIO)
};

}

#endif // CIML_GENETIC_H
