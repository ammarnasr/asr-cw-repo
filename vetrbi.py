import observation_model
import math
import openfst_python as fst
import time
from utils import parse_lexicon, generate_symbol_tables
import numpy as np

class MyViterbiDecoder:
    
    NLL_ZERO = 1e10  # define a constant representing -log(0).  This is really infinite, but approximate
                     # it here with a very large number
    
    def __init__(self, f, audio_file_name, verbose=False, use_pruning=False, determinized=False, bigram = False, histogram_pruning_threshold = 0):
        """Set up the decoder class with an audio file and WFST f
        """
        self.lex = parse_lexicon('lexicon.txt')
        self.verbose = verbose
        self.om = observation_model.ObservationModel()
        self.f = f
        self.number_of_computiations = 0
        self.decode_time = 0
        self.backtrace_time = 0
        self.use_pruning = use_pruning
        self.prune_threshold = self.NLL_ZERO
        self.determinized = determinized
        self.bigram = bigram
        self.word_start = -1
        self.histogram_pruning_threshold = histogram_pruning_threshold


        for state in self.f.states():
            for arc in self.f.arcs(state):
                o = self.f.output_symbols().find(arc.olabel)
                if o =='sil':
                    self.word_start = arc.nextstate
                   
                    

        if self.use_pruning:
            self.prune_threshold = 30.0
        
        if audio_file_name:
            self.om.load_audio(audio_file_name)
        else:
            self.om.load_dummy_audio()
        
        self.initialise_decoding()

        
    def initialise_decoding(self):
        """set up the values for V_j(0) (as negative log-likelihoods)
        
        """
        
        self.V = []   # stores likelihood along best path reaching state j
        self.B = []   # stores identity of best previous state reaching state j
        self.W = []   # stores output labels sequence along arc reaching j - this removes need for 
                      # extra code to read the output sequence along the best path
        
        for t in range(self.om.observation_length()+1):
            self.V.append([self.NLL_ZERO]*self.f.num_states())
            self.B.append([-1]*self.f.num_states())
            self.W.append([[] for i in range(self.f.num_states())])  #  multiplying the empty list doesn't make multiple
        
        # The above code means that self.V[t][j] for t = 0, ... T gives the Viterbi cost
        # of state j, time t (in negative log-likelihood form)
        # Initialising the costs to NLL_ZERO effectively means zero probability    
        
        # give the WFST start state a probability of 1.0   (NLL = 0.0)
        self.V[0][self.f.start()] = 0.0
        
        # some WFSTs might have arcs with epsilon on the input (you might have already created 
        # examples of these in earlier labs) these correspond to non-emitting states, 
        # which means that we need to process them without stepping forward in time.  
        # Don't worry too much about this!  
        self.traverse_epsilon_arcs(0)        


    def traverse_epsilon_arcs(self, t):
        """Traverse arcs with <eps> on the input at time t
        
        These correspond to transitions that don't emit an observation
        
        We've implemented this function for you as it's slightly trickier than
        the normal case.  You might like to look at it to see what's going on, but
        don't worry if you can't fully follow it.
        
        """
        
        states_to_traverse = list(self.f.states()) # traverse all states
        while states_to_traverse:
            
            # Set i to the ID of the current state, the first 
            # item in the list (and remove it from the list)
            i = states_to_traverse.pop(0)   
        
            # don't bother traversing states which have zero probability
            if self.V[t][i] == self.NLL_ZERO:
                    continue
        
            for arc in self.f.arcs(i):
                
                if arc.ilabel == 0:     # if <eps> transition
                  
                    j = arc.nextstate   # ID of next state  
                
                    if self.V[t][j] > self.V[t][i] + float(arc.weight):
                        
                        # this means we've found a lower-cost path to
                        # state j at time t.  We might need to add it
                        # back to the processing queue.
                        self.V[t][j] = self.V[t][i] + float(arc.weight)
                        
                        # save backtrace information.  In the case of an epsilon transition, 
                        # we save the identity of the best state at t-1.  This means we may not
                        # be able to fully recover the best path, but to do otherwise would
                        # require a more complicated way of storing backtrace information
                        self.B[t][j] = self.B[t][i] 
                        
                        # and save the output labels encountered - this is a list, because
                        # there could be multiple output labels (in the case of <eps> arcs)
                        if arc.olabel != 0:
                            self.W[t][j] = self.W[t][i] + [arc.olabel]
                        else:
                            self.W[t][j] = self.W[t][i]
                        
                        if j not in states_to_traverse:
                            states_to_traverse.append(j)


    def forward_step(self, t):
        '''
        Propagate the Viterbi costs forward one time step
        '''
        for i in self.f.states():
            
            if not self.V[t-1][i] == self.NLL_ZERO:   # no point in propagating states with zero probability
                
                for arc in self.f.arcs(i):
                    
                    
                    if arc.ilabel != 0: # <eps> transitions don't emit an observation
                        self.number_of_computiations += 1
                        j = arc.nextstate
                        tp = float(arc.weight)  # transition prob
                        ep = -self.om.log_observation_probability(self.f.input_symbols().find(arc.ilabel), t)  # emission negative log prob


                        prob = tp + ep + self.V[t-1][i] # they're logs

                        

                        if prob < self.V[t][j]:
                            self.V[t][j] = prob
                            self.B[t][j] = i
                            
                            # store the output labels encountered too
                            if arc.olabel !=0:
                                self.W[t][j] = [arc.olabel]
                            else:
                                self.W[t][j] = []

        if self.use_pruning and self.histogram_pruning_threshold == 0:    
            best_path = min(self.V[t])
            for idx, path in enumerate(self.V[t]):
                if path - best_path > self.prune_threshold:
                    self.V[t][idx] = self.NLL_ZERO

        if self.use_pruning and self.histogram_pruning_threshold > 0:
            # get the indices and values of paths in V[t] that do not have NLL_ZERO probability
            indices = [i for i, x in enumerate(self.V[t]) if x != self.NLL_ZERO]
            values = [x for x in self.V[t] if x != self.NLL_ZERO]

            # check if there are more than histogram_threshold paths
            if len(values) > self.histogram_pruning_threshold:
                # keep the histogram_threshold best paths and set the rest to NLL_ZERO
                best_paths = np.argpartition(values, self.histogram_pruning_threshold)[:self.histogram_pruning_threshold]
                for idx in indices:
                    if idx not in best_paths:
                        self.V[t][idx] = self.NLL_ZERO
            
                         
    
    def finalise_decoding(self):
        """ this incorporates the probability of terminating at each state
        """
        
        for state in self.f.states():
            final_weight = float(self.f.final(state))
            if self.V[-1][state] != self.NLL_ZERO:
                if final_weight == math.inf:
                    self.V[-1][state] = self.NLL_ZERO  # effectively says that we can't end in this state
                else:
                    self.V[-1][state] += final_weight
                    
        # get a list of all states where there was a path ending with non-zero probability
        finished = [x for x in self.V[-1] if x < self.NLL_ZERO]
        if not finished and self.verbose:  # if empty
            print("No path got to the end of the observations.")
        
        
    def decode(self):
        start = time.time()
        self.initialise_decoding()
        t = 1
        while t <= self.om.observation_length():
            self.forward_step(t)
            self.traverse_epsilon_arcs(t)
            t += 1
            
        self.finalise_decoding()
        end = time.time()
        self.decode_time = end-start
        if self.verbose:
            print("Decoding took", self.decode_time,  "seconds")
            print("Number of computations:", self.number_of_computiations)

    
    def backtrace(self):
        start = time.time()
        best_final_state = self.V[-1].index(min(self.V[-1])) # argmin
        best_state_sequence = [best_final_state]
        best_out_sequence = []
        
        t = self.om.observation_length()   # ie T
        j = best_final_state


        next_i = -1
        prev_next_i = next_i

 
        while t >= 0:
            i = self.B[t][j]
            best_state_sequence.append(i)
            if i == self.word_start and self.determinized:
                best_out_sequence = self.W[t][j] + [0] + best_out_sequence
            elif self.determinized: 
                best_out_sequence = self.W[t][j] + best_out_sequence                              

            if i >=0 and not self.determinized:
                for arc in self.f.arcs(i):
                    next_i = arc.nextstate
                    if next_i != i and next_i != prev_next_i:
                        if float(self.f.final(next_i)) != math.inf:
                            best_out_sequence = self.W[t][j] + [0] + best_out_sequence
                            prev_next_i = next_i
                        else:
                            best_out_sequence = self.W[t][j] + best_out_sequence


            

            # continue the backtrace at state i, time t-1
            j = i  
            t-=1
            
        best_state_sequence.reverse()
        
        
        best_out_sequence = ' '.join([ self.f.output_symbols().find(label) for label in best_out_sequence])

        if self.bigram:
            split_word = "sil"
        else:
            split_word = "<eps>"

        phones = best_out_sequence.split(split_word)
        phones[0] = phones[0].replace("sil","")
        print('Phones: ', phones)
        print('best_out_sequence: ', best_out_sequence)
        print('best_state_sequence: ', best_state_sequence)
        words = ""

        for phone in phones:
            phone = phone.replace(' ', '')
            for key, val in self.lex.items():
                val = ''.join(val)
                if val == phone:
                    words += key + " "


        end = time.time()
        self.backtrace_time = end-start
        if self.verbose:
            print("Backtrace took", self.backtrace_time, "seconds")
        
        return (best_state_sequence, words)
