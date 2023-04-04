import openfst_python as fst
import math
import pickle
from utils import parse_lexicon, generate_symbol_tables
from utils import get_unigram_and_bigram_probs
unigram_probs, bigram_probs = get_unigram_and_bigram_probs()

lex = parse_lexicon('lexicon.txt')
word_table, phone_table, state_table = generate_symbol_tables(lex)




def generate_phone_wfst(f, start_state, phone, n):
    """
    Generate a WFST representating an n-state left-to-right phone HMM
    
    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assmed to exist already
        phone (str): the phone label 
        n (int): number of states for each phone HMM
        
    Returns:
        the final state of the FST
    """


    current_state = start_state
    
    for i in range(1, n+1):
        
        in_label = state_table.find('{}_{}'.format(phone, i))
        
        sl_weight = fst.Weight('tropical', -math.log(0.1))  # weight for self-loop
        # self-loop back to current state
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))
        
        # transition to next state
        
        # we want to output the phone label on the final state
        # note: if outputting words instead this code should be modified
        if i == n:
            out_label = phone_table.find(phone)
        else:
            out_label = 0   # output empty <eps> label
            
        next_state = f.add_state()
        next_weight = fst.Weight('tropical', -math.log(0.9)) # weight to next state
        f.add_arc(current_state, fst.Arc(in_label, out_label, next_weight, next_state))    
       
        cp = current_state
        current_state = next_state

    return current_state


def generate_word_wfst(word):
    """ Generate a WFST for any word in the lexicon, composed of 3-state phone WFSTs.
        This will currently output word labels.  
        Exercise: could you modify this function and the one above to output a single phone label instead?
    
    Args:
        word (str): the word to generate
        
    Returns:
        the constructed WFST
    
    """
    f = fst.Fst('log')
    
    # create the start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    current_state = start_state
    
    # iterate over all the phones in the word
    for phone in lex[word]:   # will raise an exception if word is not in the lexicon
        
        current_state = generate_phone_wfst(f, current_state, phone, 3)
    
        # note: new current_state is now set to the final state of the previous phone WFST
        
    f.set_final(current_state)
    
    return f


def generate_word_sequence_recognition_wfst(n):
    """ generate a HMM to recognise any single word sequence for words in the lexicon
    
    Args:
        n (int): states per phone HMM

    Returns:
        the constructed WFST
    
    """
    
    f = fst.Fst()
    
    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    
    for word, phones in lex.items():
        current_state = f.add_state()
        f.add_arc(start_state, fst.Arc(0, 0, None, current_state))
        
        for phone in phones: 
            current_state = generate_phone_wfst(f, current_state, phone, n)
        # note: new current_state is now set to the final state of the previous phone WFST
        
        f.set_final(current_state)
        f.add_arc(current_state, fst.Arc(0, 0, None, start_state))
        
    return f

def generate_word_sequence_recognition_wfst_with_silance(n, use_unigram_probs=False):
    """ generate a HMM to recognise any single word sequence for words in the lexicon with 2 silence states at the beginning and end
    
    Args:
        n (int): states per phone HMM

    Returns:
        the constructed WFST
    
    """
    if use_unigram_probs:
        with open('unigram_probs.pickle', 'rb') as handle:
            unigram_probs = pickle.load(handle)


    f = fst.Fst()

    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)

    silence_state = generate_phone_wfst(f, start_state, 'sil', 3)
    i = 0
    for word, phones in lex.items():
        
        
        current_state = f.add_state()
        if use_unigram_probs:
            unigram_weight = fst.Weight('tropical', -math.log(unigram_probs[word]))
        else:
            unigram_weight = None
        f.add_arc(silence_state, fst.Arc(0, 0, unigram_weight , current_state))

        for index, phone in enumerate(phones):

            current_state = generate_phone_wfst(f, current_state, phone, n)
        # note: new current_state is now set to the final state of the previous phone WFST

        f.set_final(current_state)
        f.add_arc(current_state, fst.Arc(0, 0, None, silence_state))



    return f




def generate_bigram_wfst(n):
    """ generate a HMM to recognise any single word sequence for words in the lexicon with 2 silence states at the beginning and end
    
    Args:
        n (int): states per phone HMM

    Returns:
        the constructed WFST
    
    """


    f = fst.Fst()

    # create a single start state
    start_state = f.add_state()
    f.set_start(start_state)
    silence_state = generate_phone_wfst(f, start_state, 'sil', 3)



    word_states_dict = {}
    for word, phones in lex.items():
        word_state = f.add_state()
        bigram_start_weight = fst.Weight('tropical', 1e10)
        if ('<start>', word) in bigram_probs:
            bigram_start_weight = fst.Weight('tropical', -math.log(bigram_probs[('<start>',word)]))

        f.add_arc(silence_state, fst.Arc(0, 0, bigram_start_weight, word_state))
        word_states_dict[word] = word_state
        f.set_final(word_state)
        
    for key, value in word_states_dict.items():
        for word, phones in lex.items():
            current_state = f.add_state()


            bigram_weight = fst.Weight('tropical', 1e10)
            
            if (key, word)  in bigram_probs:
                bigram_weight = fst.Weight('tropical', -math.log(bigram_probs[(key,word)]))

            f.add_arc(value, fst.Arc(0, 0, bigram_weight, current_state))
            for index, phone in enumerate(phones):

                current_state = generate_phone_wfst(f, current_state, phone, n)

            silence_state = generate_phone_wfst(f, current_state, 'sil', 3)

            f.add_arc(silence_state, fst.Arc(0, 0, None, word_states_dict[word]))
    return f


