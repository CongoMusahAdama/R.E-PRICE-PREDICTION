from pydantic import BaseModel

class PropertyPricePred(BaseModel):
    PropertyType: float
    ClubHouse: float
    School_University_in_Township: float
    Hospital_in_Township: float
    Mall_in_Township: float
    Park_Jogging_track: float
    Swimming_Pool: float
    Gym: float
    Property_Area_in_VEE_estate_ft: float
    Price_by_sub_area: float
    Amenities_score: float
    Price_by_Amenities_score: float
    Noun_Counts: float
    Verb_Couts: float
    Adjective_Counts: float
    boasts_elegent: float
    elegent_towers: float
    every_day: float
    great_community: float
    mantra_gold: float
    offering_bedroom: float
    quality_specification: float
    stories_offering: float
    towers_stories: float
    world_class: float 