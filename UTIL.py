from operator import itemgetter
from obspy.core.util import AttribDict
from obspy import read, read_events, read_inventory
from obspy.geodetics import gps2dist_azimuth
from mpl_toolkits.basemap import Basemap
from pyproj import Geod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.imaging.mopad_wrapper import beach
from pyrocko.moment_tensor import MomentTensor, symmat6

def make_stats(event=None, station=None):
    """
    Map event and station object to stats with attributes.

    :param event: ObsPy `~obspy.core.event.event.Event` object
    :param station: station object with attributes latitude, longitude and
        elevation
    :return: ``stats`` object with station and event attributes
    """
    ## support function 1
    def __get_event_origin_prop(h):
        def wrapper(event):
            try:
                r = (event.preferred_origin() or event.origins[0])[h]
            except IndexError:
                raise ValueError('No origin')
            if r is None:
                raise ValueError('No origin ' + h)
            if h == 'depth':
                r = r / 1000
            return r
        return wrapper
    ## support function 2
    def __get_event_magnitude(event):
        try:
            return (event.preferred_magnitude() or event.magnitudes[0])['mag']
        except IndexError:
            raise ValueError('No magnitude')
    ## support function 3
    def __get_event_id(event):
        evid = event.get('resource_id')
        if evid is not None:
            evid = str(evid)
        return evid
    ## cataloge 1
    _STATION_GETTER = (('network', itemgetter('network')),
                       ('station', itemgetter('station')),
                       ('location', itemgetter('location')),
                       ('channel', itemgetter('channel')),
                       ('station_latitude', itemgetter('latitude')),
                       ('station_longitude', itemgetter('longitude')))
    _PANDASHEADER_GETTER = (('network', itemgetter('NET')),
                            ('station', itemgetter('STA')),
                            ('location', itemgetter('LOC')),
                            ('channel', itemgetter('CHA')),
                            ('station_latitude', itemgetter('LAT')),
                            ('station_longitude', itemgetter('LON')),
                            ('is_OK', itemgetter('OK')))
    ## cataloge 2
    _EVENT_GETTER = (
        ('event_latitude', __get_event_origin_prop('latitude')),
        ('event_longitude', __get_event_origin_prop('longitude')),
        ('event_depth', __get_event_origin_prop('depth')),
        ('event_magnitude', __get_event_magnitude),
        ('event_time', __get_event_origin_prop('time')),
        ('event_id', __get_event_id))
    ## process
    stats = AttribDict({})
    if event is not None:
        for key, getter in _EVENT_GETTER:
            stats[key] = getter(event)
    if station is not None:
        try:
            for key, getter in _STATION_GETTER:
                stats[key] = getter(station)
        except KeyError:
            for key, getter in _PANDASHEADER_GETTER:
                stats[key] = getter(station)
    ## calculate distance, azimul, backazimuth (from event to station)
    dist, az, baz = gps2dist_azimuth(stats.event_latitude,
                                    stats.event_longitude,
                                    stats.station_latitude,
                                    stats.station_longitude)
    stats.update({'distance': dist/1e3, 'azimuth': az, 'back_azimuth': baz})
    return stats

def prepare_stats(event, config_fname, window_params):
    ## event time
    event_time = (event.preferred_origin() or event.origins[0])['time']
    event_depth = (event.preferred_origin() or event.origins[0])['depth']
    ## form a list of STATS objects from event and input configuration file
    objstats = []
    fp = pd.read_csv(config_fname, header=0)
    for key, row in fp.iterrows():
        ## create a station object having a name, latitude and longitude
        s = make_stats(event, row)
        s.update(window_params)
        ## add all station to object dict so the Green's function need not to be re-computed
        objstats.append(s)
    ## return station objects sorted by their epicentra distances
    return sorted(objstats, key=lambda obj: obj.distance)

def update_with_obsdata(objstats, station_fname, waveform_fname, filter_params, 
                            delta=1, waveform_pad=30):
    ## read inventory file
    inv = read_inventory(station_fname) if station_fname is not None else None
    data_stream = read(waveform_fname)
    new_objstats = []
    for obj in objstats:
        if not obj['is_OK']: continue
        ## read mseed file
        st = data_stream.select(network=obj.network, station=obj.station, channel=obj.channel+'?')
        # print (st[0].stats)
        ## cut window with appropriate padding for pre-processing
        t1 = obj.event_time + (0 if obj.vred<=0 else obj.distance/obj.vred) + obj.t0 - waveform_pad
        t2 = t1 + obj.window + 2*waveform_pad
        st.trim(t1, t2, nearest_sample=False)
        ## remove instrumental response
        if inv is not None: st.remove_response(inventory=inv, output='VEL')
        ## filter and resample waveforms
        st.detrend('linear')
        st.taper(max_percentage=0.1)
        st.filter('bandpass', **filter_params)
        st.resample(1/delta)
        ## after filtering, now cut the waveform in the correct window
        t1 = obj.event_time + (0 if obj.vred<=0 else obj.distance/obj.vred) + obj.t0
        t2 = t1 + obj.window
        st.trim(t1, t2, nearest_sample=False)
        ## append data into output array
        npts = int(obj.window / delta)
        try:
            data = [st.select(component=c)[0].data for c in 'ZNE']
            if np.any([len(d) != npts for d in data]): raise Exception
            ## update the station object with observed data
            obj.update({'obsdata':np.array(data)})
            new_objstats.append(obj)
        except Exception as ex: continue
    return new_objstats

def plot_lune_frame(ax, frame_color='k', grid_color='lightgray', fontweight='bold',
                    clvd_left=True, clvd_right=True, lon_0=0):
    g = Geod(ellps='sphere')
    bm=Basemap(projection='hammer',lon_0=lon_0)
    ## Make sure that the axis has equal aspect ratio
    ax.set_aspect('equal')
    ## Plot meridian grid lines
    lats = np.arange(-90, 91)
    for lo in range(-30, 31, 10):
        lons = np.ones(len(lats)) * lo
        x, y = bm(lons, lats)
        ax.plot(x, y, lw=0.5, c=grid_color)
    x0, y = bm(-30*np.ones(len(lats)), lats)
    x1, y = bm(30*np.ones(len(lats)), lats)
    # ax.fill_betweenx(y, x0, x1, lw=0)
    ## Plot the left most meridian boundary
    lons = np.ones(len(lats)) * -30
    x, y = bm(lons, lats)
    ax.plot(x, y, lw=1, color=frame_color)
    ## Plot the right most meridian boundary
    lons = np.ones(len(lats)) * 30
    x, y = bm(lons, lats)
    ax.plot(x, y, lw=1, c=frame_color)
    ## Plot parallel grid lines
    lons = np.arange(-30, 31)
    for la in range(-90, 91, 10):
        lats = np.ones(len(lons)) * la
        x, y = bm(lons, lats)
        ax.plot(x, y, lw=0.5, c=grid_color)
    ## Put markers on special mechanism
    #-- isotropic points
    x, y = bm(0, 90)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    ax.annotate('+ISO', xy=(x, y*1.03), fontweight=fontweight, ha='center')
    ax.plot(x, 0, 'o', c=frame_color, ms=2)
    ax.annotate('-ISO', xy=(x, -y*.03), fontweight=fontweight, ha='center', va='top')
    #-- CLVD points
    x, y = bm(30, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    if clvd_right: ax.annotate('+CLVD', xy=(1., 0.5), xycoords='axes fraction', fontweight=fontweight, \
                rotation='vertical', va='center')
    x, y = bm(-30, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    if clvd_left: ax.annotate('-CLVD', xy=(0, 0.5), xycoords='axes fraction', fontweight=fontweight, 
                rotation='vertical', ha='right', va='center')
    # -- Double couple point
    x, y = bm(0, 0)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    ax.annotate('DC', xy=(x, y*1.03), fontweight=fontweight, \
                ha='center', va='bottom')
    # -- LVD
    lvd_lon = 30
    lvd_lat = np.degrees(np.arcsin(1/np.sqrt(3)))
    x, y = bm(-lvd_lon, lvd_lat)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    x, y = bm(lvd_lon, 90-lvd_lat)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    arc = g.npts(-lvd_lon, lvd_lat, lvd_lon, 90-lvd_lat, 50)
    x, y = bm([p[0] for p in arc], [p[1] for p in arc])
    ax.plot(x, y, lw=1, c=frame_color)

    x, y = bm(-lvd_lon, lvd_lat-90)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    x, y = bm(lvd_lon, -lvd_lat)
    ax.plot(x, y, 'o', c=frame_color, ms=2)
    arc = g.npts(-lvd_lon, lvd_lat-90, lvd_lon, -lvd_lat, 50)
    x, y = bm([p[0] for p in arc], [p[1] for p in arc])
    ax.plot(x, y, lw=1, c=frame_color)
               
    return bm
    
def _mt2lune(mxx, myy, mzz, mxy, mxz, myz):
    m33 = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]])
    eivals = np.linalg.eigvals(m33)
    eivals.sort()
    
    ## lune longitude calculated from the eigen value triple
    nom = -eivals[0] - eivals[2] + 2 * eivals[1]
    den = np.sqrt(3) * (eivals[2]- eivals[0])
    gamma = np.arctan2(nom, den) / np.pi * 180

    ## lune latitude calculated from the eigen value triple
    nom = np.sum(eivals)
    den = np.sqrt(3) * np.sqrt(np.sum(eivals**2))
    beta = np.arccos(nom / den) / np.pi * 180

    ## orientation angles determined from the eigen vector triple
    return gamma, 90 - beta
mt2lune = np.vectorize(_mt2lune)
        
def plot_summary(history_log, model_name=None, delta=1, previous_solutions=None):
    if type(history_log) is str:
        with open(history_log, 'rb') as fp: history_log = pickle.load(fp)
    elif type(history_log) is not dict:
        raise Exception('Type of `history_log` must be either string or dictionary.')
    
    objstats = history_log['stations']
    ns = len(objstats)
    nc, ne, nt = objstats[0]['Gtensor'].shape
    evdp_in_km = objstats[0].event_depth
    ## gather data to creat Gtensor
    Gtensor = np.array([_['Gtensor'] for _ in objstats])
    Obs =     np.array([_['obsdata'] for _ in objstats])    ## optimal MT ans time shift solution
    ## solution obtained
    idx = np.argmin(history_log['loss'])
    Msol = history_log['M'][idx] # MT solution is in North, East, Down coordinate system
    tshift = -history_log['t'][idx]
    # M6 east-north-up; Mt in Nm because Green tensors was calculated at 1e20 dyne-cm
    mt_sol = MomentTensor(symmat6(*Msol)*1e15) 
    ## ISO/DC/CLVD decomposition
    decomp = mt_sol.standard_decomposition()
    ## optimal waveform fit
    pred = Msol @ Gtensor  
    ## plotting auxillary variables
    scale = 1/np.max(np.abs(Obs))
    tvec = np.arange(nt) * delta
    ## output text
    txt = ('Model: %s\nDepth: %.1f km\nMw: %.1f\n' % 
           (model_name, evdp_in_km, mt_sol.moment_magnitude()))
    txt += 'ISO/DC/CLVD: %.0f/%.0f/%.0f\n' % (decomp[0][1]*100, decomp[1][1]*100, decomp[2][1]*100)

    ## setting up figure layout
    fig = plt.figure(figsize=(7.1, 8.5))
    gs0 = fig.add_gridspec(2, 1, height_ratios=(1, 2.5))
    gs1 = gs0[0].subgridspec(1, 3, width_ratios=(.9, .7, 1))
    gs2 = gs0[1].subgridspec(1, 4, width_ratios=(1, 1, 1, 0.7))
    ## evolution of L2 cost function
    ax = fig.add_subplot(gs1[2])
    ax.annotate('C', xy=(-.1, 1), xycoords='axes fraction', ha='right', va='top', fontsize=14)
    ax.semilogy(history_log['loss'], color='k')
    ax.set(xlabel='Epoch', ylabel='Cost')
    ax.annotate(txt, xy=(0.95, .95), xycoords='axes fraction', va='top', ha='right')
    ## lune diagram of source type
    ax = fig.add_subplot(gs1[1])
    ax.annotate('B', xy=(0, 1), xycoords='axes fraction', ha='right', va='top', fontsize=14)
    ax.set(frame_on=False, xticks=[], yticks=[])
    bm = plot_lune_frame(ax, fontweight=None)
    if previous_solutions is not None:
        for name, mt6 in previous_solutions.items():
            x, y = bm(*mt2lune(*mt6))
            ax.plot(x, y, 'o', mec='k', mew=.5, label=name, ms=5)
    # this study's solution
    x, y = bm(*mt2lune(*mt_sol.m6()))
    ax.plot(x, y, 'X', c='r', mec='k', mew=.5, label='This study', ms=5)
    ax.legend(loc='lower center', fontsize=6)
    ## map pf station 
    ax = fig.add_subplot(gs1[0])
    ax.annotate('A', xy=(-.1, 1), xycoords='axes fraction', ha='right', va='top', fontsize=14)
    bm = Basemap(projection='merc',llcrnrlat=35.5,urcrnrlat=41,\
            llcrnrlon=-124.0,urcrnrlon=-117,lat_ts=40,resolution='l', ax=ax)
    bm.drawcoastlines(linewidth=.75)
    bm.drawstates(linewidth=.75)
    bm.fillcontinents(color='lightgray')
    parallels = np.arange(34.,41.,2.)
    bm.drawparallels(parallels,labels=[1, 1, 0, 0], linewidth=.25, dashes=(3,2), color='gray')
    meridians = np.arange(-124, -117, 3)
    bm.drawmeridians(meridians,labels=[0, 0, 0, 1], linewidth=.25, dashes=(3,2), color='gray')
    x, y = bm([o.station_longitude for o in objstats], [o.station_latitude for o in objstats])
    ax.plot(x, y, 'v', c='skyblue', mec='k', mew=.5, ms=5)
    for i, o in enumerate(objstats): 
        if o.station in ['SAO', 'BRIB', 'BKS', 'BRK']: continue
        ax.text(x[i], y[i]+1.8e4, o.station, fontsize=6, ha='center')
    x, y = bm(objstats[0].event_longitude, objstats[0].event_latitude)
    xb, yb = bm(-118, 39)
    ax.plot([xb, x], [yb, y], lw=.5, c='k', ls='--')
    ax.plot(x, y, 'o', ms=2, c='r')
    ax.add_collection(beach(mt_sol.m6_up_south_east(), width=8e4, xy=(xb, yb), linewidth=.5, facecolor='r'))

    ## plot waveforms
    for c in range(nc):
        ax = fig.add_subplot(gs2[c])
        if c == 0: ax.annotate('D', xy=(-.1, 1), xycoords='axes fraction', ha='right', fontsize=14)
        ## plot scaled waveforms
        for s in range(ns):
            ax.plot(tvec, Obs[s, c]*scale+s, lw=1, color='k', 
                          label=(None if s>0 else 'Observation'))
            ax.plot(tvec+tshift[s], pred[s, c]*scale+s, lw=1, color='r', 
                          label=(None if s>0 else 'Solution'))
        ## annotate
        ax.set(xlabel='Time (s)', frame_on=False)
        ax.annotate(['Up', 'North', 'East'][c], xy=(0, 1), xycoords='axes fraction', va='top')
        ax.axhline(-1, lw=2, c='k')
        ax.set(xlim=(tvec[0], tvec[-1]), ylim=(-1, ns-.5))
        ax.set(yticks=range(ns), yticklabels=([obj.station for obj in objstats] if c==0 else []))
    ax.legend(loc='lower right')
    ## plot time shifts
    ax = fig.add_subplot(gs2[3])
    ax.barh(np.arange(ns), tshift-np.mean(tshift), color='r', height=0.2)
    ax.plot([0, 0], (-.1, ns-.9), lw=1, c='k')
    ax.set(yticks=[], ylim=(-1, ns-.5), frame_on=False)
    ax.axhline(-1, lw=1, c='k')
    ax.set(xlabel='Time-shift (s)')
    fig.tight_layout()
