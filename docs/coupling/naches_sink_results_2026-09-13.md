# The lateral sink at Naches, and the water the return leg was creating

Three walks of the same 1979 year, 18 columns, monthly windows, sealed bottom,
QCHARGE as the forward flux, the HAND datum (A = 40 km2) and tau = 42.7 d:

| walk | job | return leg | wall time |
|---|---|---|---|
| `walk_naches_sealed_v2` (before) | 775061 | water table + soil profile every window, no sink | 1 h 32 min, 9 columns |
| `walk_naches_sink` | 778358 | water table + soil profile every window | 1 h 14 min |
| `walk_naches_sink_wt` | 778365 | soil profile at window 0 only, then the water table alone | 1 h 14 min |

## 1. The sink bounds every column

In the sealed walk five melt-fed columns filled to the surface by days 90 to 151
(col_10 reached 0.00 m at day 90). With the sink, all 18 columns walked the full
year in both sink walks; none reached the surface; the melt-fed columns ended the
year between 5.4 and 5.7 m (water-table-only return leg). The outflow of the
unfed deep columns decays with e-folding times of 32 to 66 d against the 42.7 d
dial (shallow-datum columns near the dial, deep bedrock bands slower).

## 2. The profile stamp was a pump

ELM's own water balance in the profile-stamped sink walk, col_17: precipitation
1191 mm; drainage 3262 mm; recharge 2330 mm; runoff 198 mm; ET 228 mm; storage
change -5 mm. About 2,500 mm appeared from nowhere. Within each window ELM's
storage fell 150 to 370 mm and jumped back at the window boundary, where the
return leg copies PFLOTRAN's near-saturated soil profile into ELM's soil. ELM
drained that water down as recharge, the stamp refilled it, the sink removed it.
Recharge climbed from 3.4 mm/day in January to 7.4 mm/day in December with no
melt signal, twice the column's precipitation. The sealed walk had the same
signature (986 mm of drainage in the March window before col_17 overflowed), so
the "amplifier" seen in August was largely this, not physics.

## 3. Water table only after window 0: the year closes

Same setup, `--return wt`. col_17's recharge by month (mm/day):
0.49, 0.27, 0.18 through winter, 1.78 in May, **12.2 in June (the melt)**, then
0.5 to 0.8. Annual: P 1191, ET 268, runoff 153, drainage 715, recharge 653,
storage -64; residual +120 mm (10 percent) of which -90 mm is the sink's
removal mirrored into ELM's aquifer by the stamp. col_16: residual +27 mm
(2 percent). The forward flux sent to PFLOTRAN fell from 2330 to 653 mm at
col_17 and from 1681 to 535 mm at col_16, and the sink still held them.

Columns that received nothing show a positive residual matching a negative
stamp jump (col_01: +1618 mm against -1336 mm; col_05: +1321 against -1096):
that is the sink's drain-down mirrored into ELM, the exchange working. A few
columns (col_02, 03, 06, 13, 15) keep a positive residual of 26 to 54 percent
of precipitation with small jumps: water leaving through a flux the older
history files do not carry (lake or wetland runoff, capped snow); new cases now
record QRUNOFF, QRGWL and QSNWCPICE and the guard closes on QRUNOFF when present.
A NEGATIVE residual is the creation signal; none remains.

## 4. Per column, water-table-only walk against the profile-stamped twin

```
col     wt start  wt end   span   in mm clip mm   out mm efold d   | profile twin: wt end  span   | ELM year: P  residual  stamps
col_01      7.79   15.55   2.97     0.0     0.0   1617.2    36.8   |    15.55  2.97   |    316    +1618   -1336
col_02      5.69    5.76   0.12     0.0    34.0     91.3     0.0   |     5.76  0.12   |    481     +140     +56
col_03     65.49   83.80   4.99     0.0     0.0    563.9    44.3   |    83.80  4.99   |    207      +55      +6
col_05      9.28   15.65   2.08     0.0     0.0   1078.4    32.0   |    15.65  2.08   |    207    +1321   -1096
col_06     45.33   85.83  13.11     0.0     0.2   1106.3    66.1   |    85.83 13.11   |    690     +374     +10
col_07      5.47    5.77   0.14     0.0    32.6     67.4    55.5   |     5.77  0.14   |    316      +48     +26
col_08      4.43    5.66   0.65   188.1     0.5    420.5    53.5   |     5.74  0.72   |    877     +430    -284
col_09      5.00    5.62   0.38    93.5     0.0    218.7   104.5   |     5.21  0.03   |    690     +307     -45
col_10      4.39    5.56   0.89   559.9     0.0    817.8    63.2   |     5.68  0.92   |   1322     +707    -429
col_11     21.46   39.77   5.53   459.2     0.0   1499.5    48.2   |    39.91  5.62   |   1220    +2296   -1986
col_12     58.91   88.46  10.63   536.0     0.0   1296.2    64.5   |    88.46 10.63   |   1261     +533    -439
col_13     39.62   40.77   0.43     0.0     0.0     40.6    35.7   |    40.77  0.43   |    645     +249     +18
col_14     25.78   39.51   3.91   255.9     0.0    726.9    45.4   |    39.66  4.03   |    877    +1322   -1039
col_15     70.24   90.27   5.52     0.0     0.0    608.5    45.9   |    90.27  5.52   |    813     +436     +22
col_16      5.61    5.50   1.16   535.4     0.0    632.1     0.0   |     4.10  0.96   |   1260      +27     -15
col_17      4.77    5.44   1.90   653.1     0.0    787.7   152.1   |     3.79  0.90   |   1191     +120     -90
col_18     66.75   89.43   7.03   198.8     0.0    908.0    52.9   |    89.43  7.03   |    956     +398    -156
col_19     12.09   15.31   1.06   347.9     0.0   1056.4    37.9   |    15.53  1.28   |   1384    +1278    -823
```
(`in` is the forward flux after the clip policy, `out` the sink's outflow, mm
over the year; `efold` from the first two windows, 0.0 where outflow rose.)

## 5. What is settled and what is open

Settled: the outlet works and bounds the water table; the recession mapping
holds (e-folding within a few days of the dial where the band is in soil); the
return leg must stamp the water table alone after window 0, because stamping
the soil profile every window creates water; the walk now records ELM's balance
per window and warns when a year fails to close, stating the sign.

Open: the deep columns drain to the clipped datum at the column base and ELM
cannot see below 28.8 m (its aquifer clamp), so the two models disagree about
them; the datum question is answered in section 6, and HAND stays; the
positive residuals at the
columns with lake, wetland or capped-snow fractions will close once their
cases carry QRUNOFF; and the sink and return-leg code in both repositories is
still uncommitted.

Tools: `tools/walk_setup.py --return wt`, `tools/walk_sbatch.sh`,
`tools/walk_report.py WALK_DIR --against OTHER`.

## 6. The CONUS2 datum against HAND: a twin that changes ten columns

The CONUS2 twin (job 778905, `walk_naches_sink_conus2_v2`, 12 windows, 18 of 18
columns) differs from the water-table-only walk in one dial, the datum source.
The CONUS2 steady-state water table stands within 5 cm of the surface at 8 of
the 18 columns, where the deck builder refused a datum of 0.0 in the first
attempt (job 778434). The datum is therefore held at the band thickness, 1 m,
so that the seepage band keeps the thickness the conductance was mapped with,
and the clamp is recorded per column as `sink_datum_raised_to_band`. In the
remaining ten columns CONUS2 lands inside the column at col_06 (17.1 m) and
col_15 (81.9 m) and below the column base at the other eight, where both
sources are clipped to H - B and the two twins are the same experiment.

The eight clipped columns reproduce the water-table-only walk to the printed
precision. Their final water tables, spans and outflows are identical (col_11
39.77 m and 1499.5 mm out, col_17 5.44 m and 787.7 mm out, col_19 15.31 m and
1056.4 mm out), which confirms that the walk is deterministic across jobs and
that the datum source is the only difference between the twins. Column col_15
drained to its CONUS2 datum, from 70.2 m to 81.5 m with 368.6 mm of outflow and
an e-folding time of 38.1 days, against 90.3 m under HAND; the datum moved and
the timescale did not. Column col_06 received no recharge and stayed above its
17.1 m datum, so its sink never opened, while under HAND it drained to 85.8 m.

The eight columns with a surface datum never engaged the sink. Their water
tables started 4.4 to 65.5 m below the surface, where CONUS2 puts them within
5 cm of it, and none reached the top metre within the year, so the outflow was
0.0 mm in every one. Three of them received recharge and rose as sealed boxes:
col_12 from 58.9 m to 15.3 m on 533.5 mm, col_08 from 4.43 m to 2.35 m on
61.3 mm and col_10 from 4.39 m to 2.18 m on 96.5 mm, while under HAND the same
columns drained to 88.5 m, 5.66 m and 5.56 m. The other five received none,
because their QCHARGE was negative on every day (40.0 and 38.1 mm clipped at
col_02 and col_07), and their water tables moved by 0.05 to 0.36 m. The rise of
43.6 m on 533.5 mm at col_12 means the zone above its water table held almost
no drainable pore space, which the anchored spin at the mean recharge could
explain. ELM's year then fails to close with the water the stamp creates:
-690 mm at col_08, -917 mm at col_10 and -2094 mm at col_12, against stamp
jumps of +472, +503 and +1859 mm.

The datum of record stays HAND. CONUS2 and the columns' own warm start disagree
by 4 to 65 m at the valley columns, and a datum the column cannot reach within
a year is not a boundary condition but a target the column is spinning toward,
at the cost of ELM's water balance. The two sources differ by 70 m and 130 m
even at col_06 and col_15, where both lie inside the column, so no column
supports a direct check of one source against the other. Two uses of CONUS2
remain open: as the level of a multi-year spin before a walk, and as the datum
of a basin whose columns start near it. Job 778447, the first run of this
twin, lost nine finished windows to a SLURM controller timeout ("Unable to
confirm allocation"), so `elm_wrapper.run_built_case` now retries that one
signature once after 60 s; the walk resumed at window 10 as job 778905.
