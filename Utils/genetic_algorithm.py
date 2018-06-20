import numpy as np
from operator import itemgetter 
from Utils import LanguageModel
import Utils.W2V as wor2vec
import time

import random

class GeneticAlgorithm:
    
    def __init__(self, topics = [], pop_size = 100, n_iter = 10, fitness_model = None, vocab = None, cross_rate = 0.01, mut_rate = 0.2, ind_shape = (70, 100), seed = -1, verbose = 1, language_model_percentages = [1, -0.2, -1], evaluationPercentage = [0.1, 0.2, 0.3, 0.3, 0.2]):
        '''
        Genetic algorithm implementation using numpy
        '''
        self.evaluationPercentage       = evaluationPercentage
        self.topics                     = topics
        self.pop_size                   = pop_size
        self.n_iter                     = n_iter
        self.fitness_model              = fitness_model
        self.vocab                      = vocab
        self.cross_rate                 = cross_rate
        self.mut_rate                   = mut_rate
        self.ind_shape                  = ind_shape
        self.seed                       = seed
        self.verbose                    = verbose
        self.language_model_percentages = language_model_percentages
        self.language_model             = LanguageModel.LanguageModel(self.language_model_percentages)
        
        # use the w2v trained on higher number of words
        self.similarity_eval            = wor2vec.W2V(out_size = ind_shape[1], fname="Final_w2v_only_tweets")
        self.similarity_eval.load_w2v()
        
        # prints only if verbose > 0
        if self.verbose > 0:
            print("----------------------------------------------------------")
            print("Genetic algorithm with:")
            print("\t- population size: {}\n\t- number of iterations: {}\n\t- crossover rate: {}\n\t- mutation rate: {}\n\t- topics: {}".format(self.pop_size, self.n_iter, self.cross_rate, self.mut_rate, self.topics))
            print("----------------------------------------------------------")

    def start_algorithm(self):
        '''
        Fucntion that executes the algorithm 
        '''
        
        # fix the seed to make it deterministic
        if self.seed is not -1:
            np.random.seed(self.seed)

        # generate the initial population
        population = self.generate_population()
        self.test_pop = population
        
        # evaluate it
        fitness = self.evaluate_population(population)
        
        #create elapsed arrays for time measuring
        iteration_time = []
        sel_pup_time = []
        cross_pop_time = []
        mut_pup_time = []
        fitness_pup_time = []
        
        #histories
        max_fit_hist = []
        avg_fit_hist = []
        max_len_hist = []
        min_len_hist = []
        avg_len_hist = []
   
        #best so far
        best_so_far_fitness = 0 
        best_so_far = []
        best_so_far_text = ''
        # main iteration loop
    
        for i in range(self.n_iter):
            startTimeIt = time.time()
            
            print("Generation {}".format(i+1))
            
            # perform the selection of the population accordint to their fitness
            start = time.time()
            sel_population = self.select_population(population, fitness)
            sel_pup_time.append(time.time()- start)
            
            # perform the crossover of the selected population
            start = time.time()
            cros_population = self.crossover_population(sel_population)
            cross_pop_time.append(time.time()- start)
#             for i in range(3):
#                 randomTweet = random.choice(cros_population)
#                 tweetText = [x for x,_ in randomTweet]
#                 print("crossover ", i , tweetText)
            
            # mutate the crossover population
            start = time.time()
            mut_population = self.mutation_population(cros_population)
            mut_pup_time.append(time.time()- start)
#             for i in range(3):
#                 randomTweet = random.choice(mut_population)
#                 tweetText = [x for x,_ in randomTweet]
#                 print("mutation ", i , tweetText)
            
            # evaluate it
            start = time.time()
            mut_fitness = self.evaluate_population(mut_population)
            fitness_pup_time.append(time.time()- start)
            
            #gettin the lengeth
            length = np.array(self.get_length_of_individuals(mut_population))
            max_len_hist.append(np.max(length))
            min_len_hist.append(np.min(length))
            avg_len_hist.append(np.mean(length))
                 
            # combine the new population with the previous one
            population, fitness = self.combine_population(population, mut_population, fitness, mut_fitness)
            
            iteration_time.append(time.time()-startTimeIt)
            
            max_fit_hist.append(np.max(fitness))
            avg_fit_hist.append(np.mean(fitness))
            
            if  np.max(fitness)>best_so_far_fitness:
                best_so_far_fitness = np.max(fitness)
                best_so_far = population[np.argmax(fitness)]
                best_so_far_text = ' '.join([x for x,_ in best_so_far])
                
            
            # print iteration information
            if self.verbose > 0:
                print("\t- avg fitness: {}\n\t- max fitness: {}".format(np.mean(fitness), np.max(fitness)))
                print("\t- avg length:  {}\n\t- max length:  {}\n\t- min length:  {}".format(np.mean(length), np.max(length),np.min(length)))
                print("Elapsed Times: \n\t- iteration: {}s\n\t- selection: {}\n\t- crossover: {}s\n\t- mutation: {}s\n\t- evaluation: {}s".format(
                    np.mean(np.array(iteration_time)),
                    np.mean(np.array(sel_pup_time)), 
                    np.mean(np.array(cross_pop_time)), 
                    np.mean(np.array(mut_pup_time)), 
                    np.mean(np.array(fitness_pup_time)) ) )
                print("Best tweet so far with fitness: {} is: \n{}".format(best_so_far_fitness, best_so_far_text))
                print("----------------------------------------------------------")
       
    
        #return the best tweet
        tweet = population[np.argmax(fitness)]
        tweetText = ' '.join([x for x,_ in tweet])
        
        #tweetText = []
        #for tweet in population:
        #    tweetText.append(' '.join([x for x,_ in tweet]))
        
        return tweetText, np.array(max_fit_hist), np.array(avg_fit_hist), np.array(max_len_hist), np.array(min_len_hist), np.array(avg_len_hist)
        
    
    def generate_population(self):
        '''
        Function that generates a population of a given size
        with the desired representation
        '''
        
        # generate a random population using the available words in w2v
        population_words = np.random.choice(list(self.vocab.keys()), (self.pop_size, self.ind_shape[0]))
        population_init = [[[iw , self.vocab[iw][0]] for iw in ind_words] for ind_words in population_words]
        
        # cut off different sized parts of the individuals
        cut_offset = np.random.randint(0, self.ind_shape[0], self.pop_size)
        
        population_cut = [population_init[i][:cut_offset[i]] for i in range(self.pop_size)]
        
                         
        for i in range(self.pop_size):                
            #add zero tuples at the end
            for j in range(self.ind_shape[0] - cut_offset[i]):
                population_cut[i].append(  ('' , np.zeros(self.ind_shape[1], dtype = np.float32) ) )
            
            #if the there are more topics then the len of the words in the individual then fill the individual with the topics 
            offset = cut_offset[i]
            if offset < len(self.topics):
                offset = len(self.topics)
                
            #add the topics to the individuals
            if len(self.topics) > 0: 
                pos = np.random.choice(np.arange(0, offset), len(self.topics), replace =False)
                for j in range(len(self.topics)):
                    population_cut[i][pos[j]] = (self.topics[j] , self.vocab[self.topics[j]][0]) 

        self.popu = population_cut
        return population_cut
    
    def evaluate_population(self, pop):
        '''
        Function that returns the fitness of the elements of
        a given population
        '''
        #get the number of topics
        number_of_topics = len(self.topics)
        #ini the topic scores
        topic_scores = []
        if number_of_topics > 0:
            #loop through the population
            for i in range(len(pop)):
                #ini the score counter
                topic_score = 0
                #get the text in the individual
                individual = np.array([x for x,_ in pop[i]])
                #loop through the topics
                for topic in self.topics:
                    #if the topic is in the individual add 1 to the score
                    if topic in individual:
                        topic_score+=1
                #normalize and append to the topic scores list
                topic_scores.append(topic_score/number_of_topics)

            topic_scores = np.array(topic_scores[:])*self.evaluationPercentage[0]
        else:
            topic_scores = np.ones(len(pop))*self.evaluationPercentage[0]
            
        
        
        #get the language score 
        language_score = self.language_model.predict_language(pop)
        language_score = np.array(language_score)*self.evaluationPercentage[1]
          
        #get the tweet score
        pop_array = []
        for t in range(len(pop)):
            pop_array.append(np.array(np.array([np.squeeze(x) for _,x in pop[t]])))
        pop_array = np.array(pop_array)
        tweet_score = self.fitness_model.predict(pop_array)
        #get the minimum and maximum fitness that can be get
        minimum = 0
        maximum = 0
        for p in self.language_model_percentages:
            if p<0:
                minimum += abs(p)
            else:
                maximum +=p
                
        tweet_score = np.sum(np.array(tweet_score)*self.language_model_percentages, axis = 1)
        tweet_score[:] += minimum
        tweet_score[:] /= (minimum + maximum)
        tweet_score[:] *= self.evaluationPercentage[2] 
        
        # from https://www.researchgate.net/publication/283128102_Using_a_genetic_algorithm_to_produce_slogans
        diversity_score = []
        for tweet in pop:
            d = [t[0] for t in tweet if t[0] != '' ]
            if len(d)>0:
                diversity_score.append(len(set(d)) / len(d))
            else:
                diversity_score.append(0)
                
        diversity_score = np.array(diversity_score)*self.evaluationPercentage[3]
        
        similarity_score = []            
        if number_of_topics > 0:
            for tweet in pop:
                words = [t[0] for t in tweet if t[0] != '' ]
                s = []
                for to in self.topics:
                    s.append([self.similarity_eval.model.wv.similarity(to, w) for w in words])
                similarity_score.append(np.mean(np.array(s)))
            similarity_score = np.array(similarity_score)*self.evaluationPercentage[4]
        else: 
            similarity_score = np.ones(len(pop))*self.evaluationPercentage[4]
        
        #calculate the fitness
        fitness = np.sum( np.column_stack((topic_scores,language_score,tweet_score, diversity_score, similarity_score)), axis=1)
        
        return fitness
    
    def select_population(self, pop, scores):
        '''
        Function that performs tournament selection 
        '''
        
        # calculate the probabilities of each individual
        p = scores / np.sum(scores)
        indxs = np.arange(len(pop))
        sel_indxs = np.random.choice(indxs, self.pop_size, True, p)
        
        return itemgetter(*sel_indxs)(pop)
    
    def crossover_population(self, population):
        '''
        Function that performs crossover over a given population
        '''

        #if the population is uneven return an error
        if len(population)%2 !=0:
            print("the population size is uneven, len = ", len(population))
            return False

        #make new population
        NewPop = []
        #loop through the pairs in the population
        for i in range(int(len(population)/2)):
            #check if crossover is done
            crossoverDone = False
            if np.random.rand() > self.cross_rate:
                #copy the individual in pairs
                I1 = np.array([x for x,_ in population[i*2]])
                I2 = np.array([x for x,_ in population[i*2+1]])
                #in the minimum length of the text in the individuals
                minL = min(len(np.where(I1 !='')[0]), len(np.where(I2 !='')[0]))

                #if there are at least 2 words do crossover
                if minL>1: 
                    #get the 2 position to do cross over for 
                    pos = np.sort(np.random.choice(np.arange(0,minL), 2, replace =False))
                    #copy the child to be the parent
                    c1 = population[i*2]
                    c2 = population[i*2+1]
                    #fill the part to be swaped
                    c1[pos[0]:pos[1]] = c2[pos[0]:pos[1]]
                    c2[pos[0]:pos[1]] = c1[pos[0]:pos[1]]
                    #append the pair back to a new population as type list
                    NewPop.append(c1)
                    NewPop.append(c2)
                    crossoverDone = True
                    
            #if no crossover is done then append the old population
            if crossoverDone != True:
                NewPop.append(population[i*2])
                NewPop.append(population[i*2+1])

        return NewPop
    
                        
    def mutation_population(self, pop):
        '''
        Function that performs mutation over a given population
        '''
        # iterate over all individuals in the population
        for individual in pop:
                        
            # get the last index containing words
            last_indx = self.ind_shape[0] - len(np.where(np.array([x for x,_ in individual]) != ''))
            
            newIn = []
            # iterate over the configurations in the individual
            for index, configuration in enumerate(individual):
                
                # decide if mutate or not this given configuration
                if np.random.rand() > self.mut_rate:
                    
                    # delete the configuration or replace (if delete dont append it to the new individual)
                    if np.random.rand() > 0.2:                        
                        #replace
                        indx = np.random.randint(len(self.vocab))
                        iw = list(self.vocab.keys())[indx]
                        newIn.append( (iw , self.vocab[iw][0]) )
                        
                #if no mutation happens and the index is a non empty (non zero) word append it to the new individual        
                elif index < last_indx:
                    newIn.append(configuration)

            #fill the individual with zero tuples until the len is 70 (self.ind_shape[0])
            NewInLen = len(newIn)
            for j in range(self.ind_shape[0] - NewInLen):
                newIn.append(  ('' , np.zeros(self.ind_shape[0], dtype = np.float32) ) )
            #set the individuals to the mutated ones
            individual = newIn
                       
        return pop
    
    def combine_population(self, population, mut_population, scores, mut_scores):
        '''
        Function that combines the old and new population into a new one
        '''
        
        # select the best 30% of the old individuals and the 70% of the new oens
        n_olds = int(len(scores) * 0.3)
        n_news = len(scores) - n_olds
        indxs_old = np.argsort(scores)[-n_olds:][::-1]
        indxs_new = np.argsort(mut_scores)[-n_news:][::-1]
        
        # assign the new population and scores
        new_population = [population[i] for i in indxs_old] + [mut_population[i] for i in indxs_new]
        new_scores = [scores[i] for i in indxs_old] + [mut_scores[i] for i in indxs_new]
        
        return new_population, np.array(new_scores)
    
    def get_length_of_individuals(self, pop):
        length=[]
        for individual in pop:
            x = np.where(np.array([x for x,_ in individual]) !='' )
            if x[0] != []:
                length.append(len(x[0]))
            else:
                length.append(0)
        return length
