/** creating new table**/
select distinct
host_id, host_name, host_since,host_is_superhost,host_total_listings_count
into 
host
from
listings_summary;

select
id as list_id,
name as list_name,
host_id,
neighbourhood
into list
from listings_summary;

select 
host_id, id as list_id, property_type,room_type
into
property
from listings_summary;

select
id as list_id,bedrooms,bed_type,amenities,bathrooms,square_feet
into
room
from
listings_summary;

select
id as list_id,price,security_deposit,cleaning_fee,extra_people
into roomprice
from listings_summary;

select 
id as list_id,guests_included, minimum_nights,instant_bookable
into roomadddetail
from listings_summary;

select 
id as list_id, longitude, latitude,neighbourhood
into location
from listings;

select 
year, location as neighbourhood, robbery,agg_assault as assault,theft,from_car,bike,fire,graffiti,drugs
into crimerate
from [dbo].[Berlin_crimes]
where year = 2016;

select
listing_id as list_id,reviewer_id,reviewer_name,comments
into reviewer
from reviews_summary;


/** Data CLeaing **/
select * from crimerate
order by neighbourhood asc;

select * from neighbourhoods
order by neighbourhood asc;

select FK_column from crimerate
WHERE FK_column NOT IN
(SELECT PK_column from PK_neighbourhoods);

--standardize neighborhood name
update location
set neighbourhood = 'Alt-Treptow'
where neighbourhood = 'Alt  Treptow';

update location
set neighbourhood = 'Charlottenburg-Nord'
where neighbourhood = 'Charlottenburg Nord';

update location
set neighbourhood = 'Brunnenstraße Nord'
where neighbourhood = 'Brunnenstr. Nord';

update location
set neighbourhood = 'Brunnenstraße Süd'
where neighbourhood = 'Brunnenstr. Süd';

update location
set neighbourhood = 'Gatow/Kladow'
where neighbourhood = 'Gatow / Kladow';

update location
set neighbourhood = 'Köllnische Vorstadt/Spindlersfeld'
where neighbourhood = 'Kölln. Vorstadt/Spindlersf.';

update location
set neighbourhood = 'MV 1 - Märkisches Viertel'
where neighbourhood = 'MV 1';

update location
set neighbourhood = 'MV 2 - Rollbergsiedlung'
where neighbourhood = 'MV 2';

update location
set neighbourhood = 'Nord 2 - Waidmannslust/Wittenau/Lübars'
where neighbourhood = 'Nord 2';

update location
set neighbourhood = 'Nord 1 - Frohnau/Hermsdorf'
where neighbourhood = 'Nord 1';

update location
set neighbourhood = 'Ost 1 - Reginhardstr.'
where neighbourhood = 'Ost 1';

update location
set neighbourhood = 'Ost 2 - Alt-Reinickendorf'
where neighbourhood = 'Ost 2';

update location
set neighbourhood = 'West 1 - Tegel-Süd/Flughafensee'
where neighbourhood = 'West 1';

update location
set neighbourhood = 'West 2 - Heiligensee/Konradshöhe'
where neighbourhood = 'West 2';

update location
set neighbourhood = 'West 3 - Borsigwalde/Freie Scholle'
where neighbourhood = 'West 3';

update location
set neighbourhood = 'West 4 - Auguste-Viktoria-Allee'
where neighbourhood = 'West 4';

update location
set neighbourhood = 'West 5 - Tegel/Tegeler Forst'
where neighbourhood = 'West 5';

update location
set neighbourhood = 'Zehlendorf Südwest'
where neighbourhood = 'zehlendorf  Südwest';

update location
set neighbourhood = 'Zehlendorf Nord'
where neighbourhood = 'zehlendorf  Nord';

delete from location
where neighbourhood = 'Baumschulenweg';

delete from crimerate
where neighbourhood = 'Bezirk (Ch-Wi), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Fh-Kb), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Lb), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Mi), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Mz-Hd), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Nk), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Pk), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Rd), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Sp), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (St-Zd), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Th-Sb), nicht zuzuordnen';

delete from crimerate
where neighbourhood = 'Bezirk (Tp-Kp), nicht zuzuordnen';

--clean NULL value
update roomprice
set security_deposit = 0
where security_deposit is null;

update roomprice
set cleaning_fee = 0
where cleaning_fee is null;



/**for excel**/

select neighbourhood, sum(robbery) as TotalRobbery, sum(assault) as TotalAssault, sum(theft) as TotalTheft, sum(from_car) as TotalCar,sum(bike) as TotalBike,
sum(fire) as TotalFire ,sum(robbery + assault + theft + from_car + bike + fire) as TotalCrime from crimerate 
group by neighbourhood;



select l.neighbourhood, count(case when r.bed_type = 'Real Bed' then 1 else null end) as RealBed,
count(case when r.bed_type = 'Futon' then 1 else null end) as Futon,
count(case when r.bed_type = 'Pull-out sofa' then 1 else null end) as Sofa,
count(case when r.bed_type = 'AirBed' then 1 else null end) as AirBed,
count(case when r.bed_type = 'Couch' then 1 else null end) as Couch
from room r
join list l
on r.list_id = l.list_id
group by neighbourhood;


select l.neighbourhood, count(case when r.amenities like '%tv%' then 1 else null end) as HaveTv,
count(case when r.amenities like '%wifi%' then 1 else null end) as HaveWifi,
count(case when r.amenities not like '%tv%' then 1 else null end) as NoTv,
count(case when r.amenities not like '%wifi%' then 1 else null end) as NoWifi
from list l
join room r
on l.list_id = r.list_id
group by neighbourhood;

select list_id, case when amenities like '%tv%' then 'Yes' else 'No' end as TV,
case when amenities like '%wifi%' then 'Yes' else 'No' end as Wifi
from room;

select l.list_id,l.neighbourhood, p.room_type, r.bedrooms ,r.bathrooms, rp.price , rp.cleaning_fee, rp.price+rp.cleaning_fee as TotalPrice
from list l
join room r on l.list_id=r.list_id
join property p on r.list_id = p.list_id
join roomprice rp on p.list_id=rp.list_id;

select l.list_id,l.neighbourhood, r.guests_included,r.minimum_nights,l1.List_name
from location l
join roomadddetail r 
on l.list_id = r.list_id
join listings1 l1
on l.list_id = l1.id;


select l.list_id,r.reviewer_id,l.neighbourhood,r.reviewer_name,comments,l1.list_name from list l
join reviewer r
on l.list_id=r.list_id
join listings1 l1
on l.list_id = l1.id;

select l.list_id, l.neighbourhood, ll.latitude, ll.longitude,ll.price
from location l
join listings ll
on l.list_id = ll.id;




select * from crimerate;
select distinct neighbourhood_cleansed from listings_summary
order by neighbourhood_cleansed asc;
select * from neighbourhoods;
select distinct neighbourhood from location
order by neighbourhood asc;










