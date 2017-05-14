#coding=UTF-8

from collections import Counter
import tensorflow as tf

def word_to_id(vocab, unk_id, word):
  """Returns the integer id of a word string."""
  if word in vocab:
    return vocab[word]
  else:
    return unk_id

def _create_vocab(captions):
  """Creates the vocabulary of word to word_id.
  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.
  Args:
    captions: A list of lists of strings.
  Returns:
    A Vocabulary object.
  """
  print("Creating vocabulary.")
  word_counts_output_file = '/Users/wangjie/Desktop/data/captions/word_count.txt'
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  min_word_count = 4
  word_counts = [x for x in counter.items() if x[1] >= min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with tf.gfile.FastGFile(word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", word_counts_output_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  # vocab = Vocabulary(vocab_dict, unk_id)

  return vocab_dict, unk_id