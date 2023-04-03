import openfst_python as fst
from subprocess import check_call
from IPython.display import Image
import glob
from tqdm import tqdm
import os
import openfst_python as fst



def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """
    
    transcription_file = os.path.splitext(wav_file)[0] + '.txt'
    
    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()
    
    return transcription


def get_unigram_and_bigram_probs():
    i = 0
    train_split = int(0.85 * len(glob.glob('/group/teaching/asr/labs/recordings/*.wav')))  # replace path if using your own audio files


    all_transcriptions = ''
    for wav_file in tqdm(glob.glob('/group/teaching/asr/labs/recordings/*.wav')):    # replace path if using your own

        transcription = read_transcription(wav_file)
        all_transcriptions += transcription + ' \n'

        i += 1
        if i == train_split:
            break

    # count unigram counts in all_transcriptions
    unigram_counts = {}
    for word in all_transcriptions.replace("\n", " ").split():
        if word in unigram_counts:
            unigram_counts[word] += 1
        else:
            unigram_counts[word] = 1




    unigram_probs = {}
    for word, count in unigram_counts.items():
        unigram_probs[word] = count / sum(unigram_counts.values())


    # save unigram probs to pickle file
    import pickle
    with open('unigram_probs.pickle', 'wb') as handle:
        pickle.dump(unigram_probs, handle)
        # load unigram probs from pickle file
    import pickle
    with open('unigram_probs.pickle', 'rb') as handle:
        unigram_probs = pickle.load(handle)

    # count unigram counts in all_transcriptions
    bigram_counts = {}
    for line in all_transcriptions.split("\n"):
        line = line.split()
        for idx, word in enumerate(line):
            if idx > 0 and (line[idx - 1], word) in bigram_counts:
                bigram_counts[(line[idx - 1], word)] += 1
            elif idx == 0 and ("<start>", word) in bigram_counts:
                bigram_counts[("<start>", word)] += 1
            elif idx == 0 and ("<start>", word) not in bigram_counts:
                bigram_counts[("<start>", word)] = 1
            else:
                bigram_counts[(line[idx - 1], word)] = 1

    # calculate bigram probabilities
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        if bigram[0] == "<start>":
            bigram_probs[bigram] = count / len(all_transcriptions.split("\n"))
        else:
            bigram_probs[bigram] = count / unigram_counts[bigram[0]]

    return unigram_probs, bigram_probs



def draw(f):
    f.draw('tmp.dot', portrait=True)
    check_call(['dot','-Tpng','-Gdpi=400','tmp.dot','-o','tmp.png'])
    return Image(filename='tmp.png')


def parse_lexicon(lex_file):
    """
    Parse the lexicon file and return it in dictionary form.
    
    Args:
        lex_file (str): filename of lexicon file with structure '<word> <phone1> <phone2>...'
                        eg. peppers p eh p er z

    Returns:
        lex (dict): dictionary mapping words to list of phones
    """
    
    lex = {}  # create a dictionary for the lexicon entries (this could be a problem with larger lexica)
    with open(lex_file, 'r') as f:
        for line in f:
            line = line.split()  # split at each space
            lex[line[0]] = line[1:]  # first field the word, the rest is the phones
    return lex

def generate_symbol_tables(lexicon, n=3, with_silence=True):
    '''
    Return word, phone and state symbol tables based on the supplied lexicon
        
    Args:
        lexicon (dict): lexicon to use, created from the parse_lexicon() function
        n (int): number of states for each phone HMM
        
    Returns:
        word_table (fst.SymbolTable): table of words
        phone_table (fst.SymbolTable): table of phones
        state_table (fst.SymbolTable): table of HMM phone-state IDs
    '''
    
    state_table = fst.SymbolTable()
    phone_table = fst.SymbolTable()
    word_table = fst.SymbolTable()
    
    # add empty <eps> symbol to all tables
    state_table.add_symbol('<eps>')
    phone_table.add_symbol('<eps>')
    word_table.add_symbol('<eps>')
    
    for word, phones  in lexicon.items():
        
        word_table.add_symbol(word)
        
        for p in phones: # for each phone
            
            phone_table.add_symbol(p)
            for i in range(1,n+1): # for each state 1 to n
                state_table.add_symbol('{}_{}'.format(p, i))

    if with_silence:
        # add silence symbols
        phone_table.add_symbol('sil')
        for i in range(1,n+1):
            state_table.add_symbol('sil_{}'.format(i))
            
    return word_table, phone_table, state_table



