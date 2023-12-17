# CulturalAnalytics-MetadataPredictions
(related repo: [CoverPredictions](https://github.com/nicobenz/CulturalAnalytics-CoverPredictions))
## What to expect in this repo
This repo contains the code for my term paper in the module Methods and Applications in Digital Humanities of the MSc Digital Humanities at Leipzig University. 
Here I will explore classifier training using metadata for the classification of genres and subgenres in music. 
For my analysis I use music/acoustic metadata of records crawled from [AcousticBrainz](https://acousticbrainz.org) along with metadata on artists, releases, genres and subgenres from [MusicBrainz](https://musicbrainz.org). 

In the related repo (linked above) I will pursue a similar project on album cover art. 
The results of both projects are supposed to be comparable, giving insight on two different approaches to the same research question.

## Outline and research questions

## Working title
tba
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
