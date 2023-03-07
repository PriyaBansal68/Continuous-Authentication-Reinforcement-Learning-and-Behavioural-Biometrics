#raw data set was taken from 
https://ieee-dataport.org/open-access/su-ais-bb-mas-syracuse-university-and-assured-information-security-behavioral-biometrics

# Keystrokes RL

## Environment setup
Tested on Ubuntu 20.04 but following should work on all major platforms.

```zsh
# Create a virtual environment
$ python3 -m venv venv

$ source venv/bin/activate
# venv\Scripts\activate (in Windows)

# Install all requirements
$ pip install -r requiements.txt
```

## Run examples
## Full dataset

$ python main.py --dataset-path dataset/full_bbmas_dataset/ --mode train --exp-name "demoexp"
$ python main.py --dataset-path dataset/full_bbmas_dataset/ --mode test --exp-name "demoexp"

## Individual users
- Train model (learn patterns) for user_id 16
    ```
    $ python main.py --dataset-path dataset/full_bbmas_dataset/  --user 11 --mode train --exp-name "demoexp"
    ```
- Test model for user_id 11 (if you find out any user is not performing well test that user again to improve results)
    ```
    $ python main.py --dataset-path dataset/full_bbmas_dataset/  --user 11 --mode test --exp-name "demoexp"
    ```

## Configuration
Setup following key parameters in the `config.json` file:
- _No_ - Number of events in an observation
- _Nh_ - Number of events to hop to move to next observation
- _num_encoder_features_ - Number of output features after encoding feature vector
- _corrupt_bad_probability_ - Probability of corrupting a given sample
- _num_episodes_ - Number of episodes to train the model
- _c_update_ - Number of episodes after which agent (DDQN) updates its target net
- _eps_start_ - Initial exploration probability
- _eps_end_ - End exploration probability
- _eps_decay_ - Decay rate of exploration probability

## Major TODOs
- Better corruption technique


## Future work
- Summary features
- Augmentation in training

## Discussion
- Language model
- Time Warping: Simulating variable typing speed by stretching or shrinking the time intervals between key presses.
- Jittering: Adding random variations to the timing of key presses to simulate different typing habits.
- Key Swap: Interchanging the keys pressed during a typing sequence to simulate different typing patterns.
- Key Insertion/Deletion: Adding or removing key presses in a sequence to simulate typing errors or variations in typing speed.
