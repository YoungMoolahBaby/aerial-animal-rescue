from aerial_animal_rescue.converters.bambi import boxes_at, pick_frames, read_tracks

GT = """\
10,1,100,200,20,10,1.00,60,1.00,Sus scrofa (Wild boar),0,0,0
20,1,110,200,30,10,1.00,60,1.00,Sus scrofa (Wild boar),0,0,0
15,2,500,500,8,8,1.00,2,0.50,Cervus elaphus (Red deer),1,2,0
"""


def test_read_tracks_sorts_key_frames_per_track():
    tracks = read_tracks(GT)
    assert [k[0] for k in tracks[1]] == [10, 20]
    assert tracks[2] == [(15, 500.0, 500.0, 8.0, 8.0, "Cervus elaphus (Red deer)")]


def test_boxes_at_interpolates_between_key_frames_only():
    tracks = read_tracks(GT)
    assert boxes_at(tracks, 10) == [("Sus scrofa (Wild boar)", 100.0, 200.0, 120.0, 210.0)]
    # frame 15: track 1 half way (left 105, width 25), track 2 on its key frame
    assert boxes_at(tracks, 15) == [
        ("Sus scrofa (Wild boar)", 105, 200, 130, 210),
        ("Cervus elaphus (Red deer)", 500.0, 500.0, 508.0, 508.0),
    ]
    assert boxes_at(tracks, 9) == []  # before the first key frame: no box
    assert boxes_at(tracks, 21) == []  # after the last


def test_pick_frames_keeps_gap_then_caps():
    assert pick_frames([0, 10, 30, 59, 60, 61, 200], min_gap=60, max_n=10) == [0, 60, 200]
    assert pick_frames(list(range(0, 1000, 100)), min_gap=1, max_n=5) == [0, 200, 400, 600, 800]


def test_longest_gap_counts_interpolated_tracks_only():
    from aerial_animal_rescue.converters.bambi import longest_gap

    tracks = read_tracks(GT)
    assert longest_gap(tracks, 15) == 10  # track 1 interpolated 10..20; track 2 on a key
    assert longest_gap(tracks, 10) == 0
