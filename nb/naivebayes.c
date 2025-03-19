// Naive implementation of naive bayes in C

#include <stdio.h>

#define VOCAB_SIZE 6
#define N_SAMPLES 4
#define NUM_CLASSES 2

// P(Spam) and P(Not Spam)
void compute_prior(double prior[], int labels[])
{
    int spam_count = 0;
    for (int i = 0; i < N_SAMPLES; i++)
    {
        if (labels[i] == 1)
        { // spam
            spam_count++;
        }
    }
    // prior[0] -> P(Not Spam)
    // prior[1] -> P( Spam)
    prior[1] = (double)(spam_count + 1) / (N_SAMPLES + 2); // laplace smoothing
    prior[0] = 1 - prior[1];
}

void compute_likelihood(int features_X[N_SAMPLES][VOCAB_SIZE], int labels[], double likelihood[NUM_CLASSES][VOCAB_SIZE])
{
    int spam_word_count[VOCAB_SIZE] = {0};
    int not_spam_word_count[VOCAB_SIZE] = {0};
    int spam_total = 0; int not_spam_total = 0;


    for (int i = 0; i < N_SAMPLES; i++){
        for(int j = 0; j < VOCAB_SIZE; j++) {
            if (labels[i] == 1) {
                spam_word_count[j] += features_X[i][j];
            } else {
                not_spam_word_count[j] += features_X[i][j];
            }
        }

        if(labels[i] == 1) spam_total++;
        else not_spam_total++;
    }

    for (int j = 0; j < VOCAB_SIZE; j++){
        // Probability of word j appearing given the email is Spam
        likelihood[1][j] = ((double)(spam_word_count[j] + 1))/(spam_total + 2); // laplace smoothing to prevent zero probabilities
        // Probability of word j appearing given the email is Not Spam
        likelihood[0][j] = ((double)(not_spam_word_count[j] + 1)) /(not_spam_total + 2);
    }
}

int predict(int test_input[], double prior[], double likelikehood[NUM_CLASSES][VOCAB_SIZE]){

    double prob_spam = prior[1];
    double prob_not_spam = prior[0];


    for (int j = 0; j < VOCAB_SIZE; j++){
        if(test_input[j] == 1){
            prob_spam *= likelikehood[1][j];
            prob_not_spam *= likelikehood[0][j];
        } else {
            prob_not_spam *(1 - likelikehood[0][j]);
            prob_spam *= (1 -likelikehood[1][j]);
        }
    }

    return (prob_not_spam > prob_spam) ? 0 : 1;
}


int main(void)
{

    char *vocabulary[VOCAB_SIZE] = {
        "win",
        "money",
        "free",
        "entry",
        "meeting",
        "project"};

    // input dataset 1-hot encoded
    int features_X[N_SAMPLES][VOCAB_SIZE] = {
        // for a given input text, check for the words from vocabulary, if it is, mark it as 1 else 0

        {1, 1, 0, 0, 0, 0}, // win money now
        {1, 0, 1, 1, 0, 1}, // win free entry to stadium
        {0, 0, 0, 0, 1, 1}, // meetings scheduled for discussing the project
        {0, 0, 0, 0, 0, 1}, // important project update
    };

    int target_Y[N_SAMPLES] = {
        1, 1, 0, 0 // spam, spam, not spam, not spam
    };

    int test_email[VOCAB_SIZE] = {0, 0, 1, 0, 1, 0}; // schedule meeting when you are free

    double prior[NUM_CLASSES] = {0};
    double likelihood[NUM_CLASSES][VOCAB_SIZE] = {0};

    compute_prior(prior, target_Y);
    compute_likelihood(features_X, target_Y, likelihood);
    int result = predict(test_email, prior, likelihood);
    printf("Test email is classified as: %s\n", result ? "Spam": "Not Spam");
    return 0;
}