import numpy as np


def identify_gender(females_gmm, males_gmm, vector):
    # female hypothesis scoring
    is_female_scores = np.array(females_gmm.score(vector))
    is_female_log_likelihood = is_female_scores.sum()
    # male hypothesis scoring
    is_male_scores = np.array(males_gmm.score(vector))
    is_male_log_likelihood = is_male_scores.sum()

    print("%10s %5s %1s" % ("+ FEMALE SCORE", ":",
          str(round(is_female_log_likelihood, 3))))
    print("%10s %7s %1s" % ("+ MALE SCORE", ":",
          str(round(is_male_log_likelihood, 3))))

    if is_male_log_likelihood > is_female_log_likelihood:
        winner = "male"
    else:
        winner = "female"
    return winner
