# CulturalAnalytics-MetadataPredictions
(related repo: [CoverPredictions](https://github.com/nicobenz/CulturalAnalytics-CoverPredictions))
## What to expect in this repo
This repo contains the code for my term paper in the module Methods and Applications in Digital Humanities of the MSc Digital Humanities at Leipzig University. 
Here I will explore classifier training using metadata for the classification of genres and subgenres in music. 
For my analysis I use acoustic mood data of records crawled from [AcousticBrainz](https://acousticbrainz.org) along with metadata on artists, releases, genres and subgenres from [MusicBrainz](https://musicbrainz.org). 

In the related repo (linked above) I will pursue a similar project on album cover art. 
The results of both projects are supposed to be comparable, giving insight on two different approaches to the same research question.

## Working title
**Pop Music is Happy and Metal is Aggressive – Or is it?** Exploring Mood Patterns in Music Genres and their Subgenres using a Machine Learning Approach



## TODO:
### Data collection and preparation
- [x] download data from [AcousticBrainz](https://acousticbrainz.org) (very large amounts of data, possibly in 2-3 TB range... how to store?)
- [x] extract relevant metadata and map to metadata from [MusicBrainz](https://musicbrainz.org) (can partially be re-used from related repo)
- [x] calculate metadata mean of records belonging to a release to make results based on album rather than record (makes it comparable to results from album cover classifications)
### Processing and analysis
- [x] sample dataset based on available covers from other repo (to use data of same releaases for comparability)
- [x] train classifiers
- [x] evaluate
- [x] visualise distance between subgenres or clustering of classes using graph network
