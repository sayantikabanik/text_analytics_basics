"""
Take the first 1000 “rating-1.0” reviews.
Summarize them to 1% (in terms of words) using gensim and send across your summary.
Also, take the first 1000 “rating-5.0” reviews.Summarize them to approximately 300 words using gensim and send across your summary.

OUTPUT
-----------
-------------rating-1.0 !% summary of words---------------
phone
phones
worked
works
working
work
battery
batteries
batterys
charge
charging
charges
charged
like
liked
likely
likes
case
cases
casing
product
products
time
timely
times
cables
cable
look
looking
looked
looks
headset
headsets
good
goodness
goods
sounding
sound
sounds
sounded
charger
chargers
use
uses
useful
iphone
iphones
quality
qualities
better
getting
screen gets
bluetooth
bluetooths
screening
screens
motorola
ear
ears
eared
amazon
amazons
plastic
plastics
connect
connected
connection
connecting
connects
connectivity
connections
device
devices
fit
fits
fitting
fitted
fitness
calls
called
calling

-------------300 words summary------------
It is compact when folded up and very light, yet feels fairly sturdy and responsive when typing.+ I purchased this keyboard so that I could start doing some serious word processing on my Treo 650, and I've been absolutely pleased!First off, the keyboard does not come with the Treo 650 driver --- you must DOWNLOAD THIS DRIVER OFF OF THINK OUTSIDE'S websiteHowever, downloading the driver is easy, and from there, you can sync it to your Treo, or email it, in an attachment, to itOnce the drive is installed, the keyboard is flawless!My biggest complaint about BlueTooth keyboards is that they lag, as you type ---- but this is not the case with the Stowaway --- every key I pressed showed up INSTANTLY upon my Treo 650So far, it has worked terrificly with my address book, calendar, and email (VersaMail)In addition, the keyboard folds together well and is EXTREMELY thin and portable ---- I don't know if you could carry it in a pocket - but a backpack, briefcase, etc will not feel any extra weight with this productlastly - these keyboards are small --- I wish the delete key was bigger (as I often make typos) but nonetheless, I'm very happy, and certainly, typing on this quicker than the phone's keyboardhighly recommended!HAPPY TYPING!
Easy to pair, easy to adjust volume, great battery life, very clear even with wind noise and city sounds around, and last but not least it has an indicator light that does not look like an emergency strobe (I hated my mot-850 for that at night it would reflect off the glass and the dash in the car or the side window and drive you insane...)All and all a 5 star unit+ I have owned several Bluetooth cell phone headsets, and the PLT 510 is the best so far.



"""
import pandas as pd
from gensim.summarization import keywords
from gensim.summarization import summarize
import spacy
nlp = spacy.load('en_core_web_sm')


def rating_01_reviews_1000(file_name):
    df_data_all = pd.read_csv(file_name)
    df_data = df_data_all[df_data_all['rating'] == 1].head(1000)
    return df_data['reviewText']


def rating_05_reviews_1000(file_name):
    df_data_all = pd.read_csv(file_name)
    df_data = df_data_all[df_data_all['rating'] == 5].head(1000)
    return df_data['reviewText']


if __name__ == '__main__':
    str_text = []
    data_df_rating_01 = rating_01_reviews_1000('output_data_cleaned.csv')
    data_df_rating_05 = rating_05_reviews_1000('output_data_cleaned.csv')

    print("1% (in terms of words)")
    words_ = keywords('+ '.join(data_df_rating_01), ratio=0.01)
    print(words_)

    print("-------------300 words summary------------")
    print(summarize('+ '.join(data_df_rating_05), word_count=300))
