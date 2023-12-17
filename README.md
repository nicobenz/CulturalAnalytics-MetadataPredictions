# CulturalAnalytics-MetadataPredictions
(related repo: [CoverPredictions](https://github.com/nicobenz/CulturalAnalytics-CoverPredictions))
## What to expect in this repo
This repo contains the code for my term paper in the module Methods and Applications in Digital Humanities of the MSc Digital Humanities at Leipzig University. 
Here I will explore classifier training using metadata for the classification of genres and subgenres in music. 
For my analysis I use music/acoustic metadata of records crawled from [AcousticBrainz](https://acousticbrainz.org) along with metadata on artists, releases, genres and subgenres from [MusicBrainz](https://musicbrainz.org). 

In the related repo (linked above) I will pursue a similar project on album cover art. 
The results of both projects are supposed to be comparable, giving insight on two different approaches to the same research question.

## Outline and research questions
In my research paper, I aim to explore the classification of musical subgenres through their metadata using machine learning algorithms. 
Music genres typically encompass various subgenres, each characterized by unique yet subtly interconnected features that align them with their overarching genre. 
However, these connecting features are nuanced and challenging to pinpoint. 
My study will investigate whether machine learning algorithms can detect statistical patterns in this metadata, both within individual subgenres and across broader genre categories. 
A key method of analysis will be examining the confusion matrix from the classification results. 
I will argue that a significant number of true positives in this matrix may indicate a statistical relationship within a subgenre. 
More importantly, the rate of false positives, especially between subgenres of the same genre, could unveil genre-spanning features. 
For instance, I anticipate a higher rate of false positives within subgenres of Metal compared to false positives between a Metal subgenre and a Hip Hop subgenre. 
If observed, this pattern could suggest the presence of distinct, genre-specific characteristics embedded in the musical metadata.

However, I haven't yet decided on the type of metadats used for my approach. 
Here is a partial list of available metadata on [AcousticBrainz](https://acousticbrainz.org):

Low level information:
- general metadata
  - sample rate
  - bit rate
  - length
  - ...
- average loudness
- bark scale bands
- equivalent rectangular bandwidth (ERB-rate scale)
- gammatone feature cepstrum coefficients (GFCC)
- high frequency content (HFC)
- mel-frequency bands
- Mel-frequency cepstral coefficients (MFCCs) 
- pitch salience
- multiple silence rates
- spectral information
- rhythm
- ...

High level information:
- danceability
- gender
- different moods as degrees
  - acoustic
  - agressive
  - electronic
  - happy
  - party
  - relaxed
  - sad
- timbre
- tonality (tonal/atonal)
- instrumental (probability value)
- ...

(High level information was computed from low level information by AcousticBrainz using [Essentia](https://essentia.upf.edu/index.html))

## Working title
Genre-Defining Features in Musical Metadata: Unveiling Common Patterns Across Subgenres with Machine Learning Classifiers

## TODO:
### Data collection and preparation
- [ ] download data from [AcousticBrainz](https://acousticbrainz.org) (very large amounts of data, possibly in 2-3 TB range... how to store?)
- [ ] extract relevant metadata and map to metadata from [MusicBrainz](https://musicbrainz.org) (can partially be re-used from related repo)
- [ ] calculate metadata mean of records belonging to a release to make results based on album rather than record (makes it comparable to results from album cover classifications)
### Processing and analysis
- [ ] sample dataset based on available covers from other repo (to use data of same releaases for comparability)
- [ ] train classifiers
- [ ] evaluate
- [ ] repeat
- [ ] ???
- [ ] profit
- [ ] visualise distance between subgenres or clustering of classes using multidimensional scaling or t-distributed stochastic neighbor embedding
