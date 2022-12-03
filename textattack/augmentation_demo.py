import nltk
from textattack.transformations import WordSwapQWERTY
from textattack.transformations import CompositeTransformation
from textattack.transformations import WordInsertionRandomSynonym, WordSwapChangeNumber, WordSwapWordNet, WordSwapEmbedding, WordSwapRandomCharacterDeletion, WordSwapQWERTY, WordSwapChangeLocation

from textattack.constraints.pre_transformation import RepeatModification
# from textattack.constraints.pre_transformation import StopwordModification
from textattack.constraints import PreTransformationConstraint

from textattack.augmentation import Augmenter
from textattack.shared.validators import transformation_consists_of_word_swaps


class StopwordModification(PreTransformationConstraint):
    """A constraint disallowing the modification of stopwords.
    	默认使用nltk的stopwords
		自定义停用词列表，数据增强时会保持停用词不变，对句子中的其他词做替换、删除等操作，如果希望某些词不变，可以加进来
	"""

    def __init__(self, stopwords=None, language="english"):
        if stopwords is not None:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(nltk.corpus.stopwords.words(language))

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        non_stopword_indices = set()
        for i, word in enumerate(current_text.words):
            if word not in self.stopwords:
                non_stopword_indices.add(i)
        return non_stopword_indices

    def check_compatibility(self, transformation):
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_swaps(transformation)


# Set up transformation using CompositeTransformation()
transformation = CompositeTransformation([WordSwapEmbedding(), WordSwapWordNet(language='eng'), WordSwapChangeNumber(), WordSwapChangeLocation()])
# Set up constraints
constraints = [RepeatModification(), StopwordModification()]
# Create augmenter with specified parameters
augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1, transformations_per_example=10)
s = 'what is this? it is a apple. my name is tom. i am china. i am 19 years old. i cannot speak english'
# Augment!
import time
start_time = time.time()
print(augmenter.augment(s))
end_time = time.time()
print(end_time - start_time)


