# Trustworthy-NLP
This is a NLP project that takes survey responses and classifies whether or not they contain a certain epistemological framework. I no longer have access to some of the files, so this code will not function as is. 
It utilizes a combination of a bag of words, word embedding, and convolutional neural networks.
Much of this code is heavily influenced by Deep Learning for NLP by Jason Brownlee.

Below is a recap and description file that I wrote to explain my progress to my superiors at the end of my time with the group.
Much of it contains abbreviations and vocabulary that would require more extensive knowledge of the project to understand.

- The files used are created using the create() function in NewDataset.py - This created the .csv files that end in _Data
- TW.py is the main NLP file
- Old_bows.py is the bag of words file that Rebeckah made
- Results_NLP.xlsx is where the statistics of the NLP is automatically stored:
- Columns: Loss, optimizer, Layers, and Notes are manually typed in
- Notes is a brief description of the method
- Layers: D=Dense(), E=Embedding(), C1D= Conv1D(),
MP1D=MaxPooling1D(), F= Flatten()
- The numbers and words like “relu” and “sigmoid” are the arguments for
that layer
- create_vocab.ipynb is the Jupyter Notebook file where the bigram vocab list was created
- TW.auto() is the function that runs the NLP
- This runs five times, with a different fold held out each time for testing
- This occasionally freezes, so there are optional keyword arguments that will allow you to avoid repeating a fold if it froze in the middle
- Specify which column is being tested in line bows_nlp()/embed_nlp() using variables defined in lines 23-29
- Specify vocab in lines 362/429
- bows_nlp() is the function that this calls if doing a simple BOW without a
CNN
- embed_nlp() is the function call for any word embedding or CNN (or both)
- DO NOT run both, it will cause an error due to the arguments that are
commented out manually depending on the NLP technique used:
- Changes to the code are needed depending on the NLP technique:
- Inputs of define_model()
- If Bag of Words only (no CNN), input is just n_words
- If Embedding (CNN or not), input is n_words, vocab_size,
input_length, matrix
- If CNN (vocab), input is n_words, vocab_size, input_length
 - Body of define_model()
- Follow lines 242, 246, 252
- Inputs, outputs, and body of prepare_data() - Inputs:
- Train_docs, test_docs, mode in ALL (also default for BOW)
- Embedding (CNN or not) + , input_length, matrix, vocab_size
- If CNN (vocab) + , input_length
- Outputs:
- Xtrain, Xtest, tokenizer in ALL (also default for BOW)
- Embedding (CNN or not) + , embedding_matrix, vocab size
AND np.asarray() for both Xtrain and Xtest
- CNN (vocab), output + , vocab_size
- Body:
- 317/318, sequences instead of matrix if embedding (CNN or
not)
- 320-322, CNN (vocab or embed)
- 324-340, If embedding (CNN or not)
- Where prepare_data() is called in line 398
- Where define_model() is called in line 408
- doc_to_line():
- Line 91
- Line 83-88 is for Bag of Bigrams

   Category
Expected Result
Consistent Results
Uncertainty
Good Methods
Peer Review
Statistics
% Positive
21.2%
59.6%
26.9%
25.9%
7.0%
12.9%
BOW
-
0.6686
0.6325
0.4274
0.4163
0.5892
Bigrams
-
0.6618
0.5851
-
-
0.5906
Embedding
-
0.5563
-
-
-
-
CNN+BOW
0.4098
0.7103
0.5756
-
-
-
CNN+Emb ed
-
0.7080
    0.6158
  -
-
0.4580
 - The blank cells are Cohen’s Kappa scores under 0.4
- Underlined is Cohen’s Kappa greater than 0.6
- The best performance was in “Consistent Results” and “Uncertainty”, which
makes sense because they had the highest number of positively coded
responses
- Across all six categories, a CNN improves the embedding method every time,
however the BOW method is only improved by a CNN for expected result and
consistent results
- Note:
- Expected result only had about half as much coded data as the other categories
- The statistics category is very interesting because it is close to having significant agreement, despite having a low percentage of positively coded responses. Since there is such a discrepancy between the Kappa score for BOW compared
            
to Embedding, my guess is that there are some words that are so clearly relating
to statistics (chi squared,etc.) that are more objective than the other categories.
- Another interesting thing was that there were very few false positives for both the
“peer review” and “good methods” categories, but a lot of missed positives.
- This would mean that the algorithm only picked up on a small portion of
the criteria used to code this as positive, and there were others that didn’t
get picked up.
- Convolutional neural networks can improve performance when it has enough
data, and it works better with more complex data preparation(embedding rather than BOW).
