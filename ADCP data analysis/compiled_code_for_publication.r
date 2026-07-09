##### 1 ##### Packages used:

library(R.matlab)
library(tidyr)
library(dplyr)
library(scales)
library(zoo)
library(lubridate)
library(ggplot2)
library(RColorBrewer)
library(viridis)
library(ggnewscale)
library(hms)
library(fuzzyjoin)
library(broom)
library(purrr)
library(ggpubr)

##### 2 ##### Processing the acoustic backscatter data and calculating the integrated echo intensity:

heat <- readMat("Trubadur_ADCP_beam5_m30sec.mat")
val <- as.data.frame(heat$a5m)
time <- as.vector(heat$tm)
time <- as.POSIXct((time - 719529) * 86400, origin = "1970-01-01", tz = "UTC")
depth <- as.vector(heat$z)

df_wide <- data.frame(
  time,
  val
)

colnames(df_wide)[-1] <- depth

heat2 <- df_wide %>%
  pivot_longer(
    cols = -time,
    names_to = "depth",
    values_to = "value"
  )

heat2$depth <- as.numeric(heat2$depth)
str(heat2)

# creating the subplot for demonstration:

sub <- heat2 %>%
  slice(which.min(
    abs(time - as.POSIXct("2018-08-29 02:00:00", tz="Europe/Berlin"))):
      which.min(
        abs(time - as.POSIXct("2018-08-31 02:00:15", tz="Europe/Berlin"))))
sub <- filter(sub, between(depth, -20.5, -9.5))

heat2slice <- filter(heat2, depth < -9 & depth > -21)
heat2slicetotval <- aggregate(value~time,heat2slice,sum) ### the pink line representing integrated echo intensity

##### 3 ##### derivative, ascent/descent events

# calculate the derivative of the integrated echo intensity:

df <- heat2slicetotval %>%
  arrange(time) %>%
  mutate(
    smooth = rollmedian(value, k = 241, fill = NA)
  )%>%
  mutate(
    d1 = c(NA, diff(smooth)),
    day = as.Date(time)
  )

# a function to detect highest/lowest deriivative for each sunset/sunrise respectively:

find_transitions <- function(day_df, threshold_factor = 0.5, min_run_len = 5) {
  
  d1 <- day_df$d1
  vals <- day_df$smooth
  times <- day_df$time
  
  thr <- sd(d1, na.rm = TRUE) * threshold_factor
  
  rising  <- d1 >  thr
  falling <- d1 < -thr
  
  r_rise <- rle(rising)
  r_fall <- rle(falling)
  
  extract_runs <- function(rle_obj, direction = "rise") {
    lengths <- rle_obj$lengths
    values  <- rle_obj$values
    starts  <- cumsum(c(1, head(lengths, -1)))
    
    run_list <- list()
    
    for (i in seq_along(values)) {
      if (values[i] && lengths[i] >= min_run_len) {
        idx_start <- starts[i]
        idx_end   <- starts[i] + lengths[i] - 1
        
        val_change <- vals[idx_end] - vals[idx_start]
        
        if (direction == "fall") val_change <- -val_change
        
        run_list[[length(run_list) + 1]] <- list(
          start_idx = idx_start,
          end_idx   = idx_end,
          time      = times[idx_start],
          change    = val_change
        )
      }
    }
    
    do.call(rbind, lapply(run_list, as.data.frame))
  }
  
  rise_runs <- extract_runs(r_rise, "rise")
  fall_runs <- extract_runs(r_fall, "fall")
  
  best_rise <- if (!is.null(rise_runs)) rise_runs[which.max(rise_runs$change), ] else NULL
  best_fall <- if (!is.null(fall_runs)) fall_runs[which.max(fall_runs$change), ] else NULL
  
  list(
    rising_best  = best_rise,
    falling_best = best_fall,
    all_rise = rise_runs,
    all_fall = fall_runs
  )
}

# apply the function:

results_evening <- df %>%
  group_by(day) %>%
  group_modify(~ {
    out <- find_transitions(.x)
    
    tibble(
      day = unique(.x$day),
      evening_onset  = out$all_rise$time,
      evening_change = out$all_rise$change
    )
  }) %>%
  ungroup() %>%
  left_join(
    df %>% select(day, time, smooth),
    by = c("day", "evening_onset" = "time")
  )

results_morning <- df %>%
  group_by(day) %>%
  group_modify(~ {
    out <- find_transitions(.x)
    
    tibble(
      day = unique(.x$day),
      morning_onset  = out$all_fall$time,
      morning_change = out$all_fall$change
    )
  }) %>%
  ungroup() %>%
  left_join(
    df %>% select(day, time, smooth),
    by = c("day", "morning_onset" = "time")
  )

# i looked at all the results, and took out the unreasonable rise and fall times. the times that were taken out were:

#fall:
# 9/21 (185, 186)
# 9/22 (193:196)
# 9/23 (202)

#rise:
# 9/25 (201)

results_morning <- results_morning[-c(185, 186, 193:196, 202),]
results_evening <- results_evening[-c(201, 123),]

results_evening <- results_evening %>% 
  group_by(day) %>%
  slice_max(evening_change) ### use this for evening data

results_morning <- results_morning %>% 
  group_by(day) %>%
  slice_max(morning_change) ### use this for morning data

evening_clean <- results_evening %>%
  rename(
    onset  = evening_onset,
    change = evening_change
  ) %>%
  mutate(type = "rise")

morning_clean <- results_morning %>%
  rename(
    onset  = morning_onset,
    change = morning_change
  ) %>%
  mutate(type = "fall")

events <- as.data.frame(dplyr::bind_rows(evening_clean, morning_clean)) ### important table

##### 4 ##### slopes:

window_minutes <- 15
delta_seconds <- 15*60

df_slopes <- events %>%
  select(day, onset, change, type
  ) %>%
  left_join(df, by = "day", relationship = "many-to-many"
  ) %>%
  dplyr::filter(abs(as.numeric(difftime(time, onset, units = "mins"))) <= window_minutes
  ) %>%
  group_by(day, type, onset
  ) %>%
  summarise(slope = coef(lm(smooth ~ time))[2],
            .groups = "drop"
  ) %>%
  left_join(df %>% select(time, smooth), by = c("onset" = "time")) %>%
  mutate(
    x_start = onset - delta_seconds,
    y_start = smooth - slope * delta_seconds,
    x_end   = onset + delta_seconds,
    y_end   = smooth + slope * delta_seconds
  ) %>%
  select(onset, smooth, slope, type, x_start, y_start, x_end, y_end)

df_slopes <- df_slopes %>%
  mutate(
    diff_time = as.numeric(difftime(x_end, x_start, units = "secs")),
    avg_rate = (y_end - y_start) / diff_time
  )

##### 5 ##### weather and wave data:

# read in weather data acquired from SMHI:

weather <- read.csv("r_weather.csv", header = T, sep = ";", dec = ",")
weather$date <- as.Date(weather$date, format = "%Y.%m.%d")
weather$time_utc <- as_hms(weather$time_utc)
weather$datetime <- as.POSIXct(paste(weather$date, weather$time_utc), format="%Y-%m-%d %H:%M:%S")+4*60*60

# calling in the adcp wave data:

w_truba <- read.csv("w_truba.csv")
w_truba$time <- as.POSIXct(w_truba$time)
str(w_truba)

# fitting 3 h of weather data "around" the ascent and descent events and averaging it:

df_slope_env_3h <- df_slopes %>% 
  difference_left_join(
    weather,
    by = c("onset" = "datetime"),
    max_dist = as.difftime(2, units = "hours")
  ) %>%
  
  mutate(dt_hours = as.numeric(difftime(datetime, onset, units = "hours"))) %>%
  group_by(onset) %>%
  
  slice_min(abs(dt_hours), n = 1) %>%
  ungroup() %>%
  
  mutate(
    target_time = datetime,
    lower = target_time - hours(1),
    upper = target_time + hours(1)
  ) %>%
  
  left_join(
    weather,
    join_by(y$datetime >= x$lower, y$datetime <= x$upper),
    suffix = c("", "_wx")
  ) %>%
  
  group_by(onset, slope, smooth, type) %>%
  summarise(
    across(
      contains("_wx"),
      ~ mean(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  )

# fitting the 30 min avergaes to the slope data (the duration of the slope):

w_truba_slope <- df_slopes %>%
  rowwise() %>%
  mutate(
    mean_wave = {
      t_start_closest <- w_truba$time[which.min(abs(w_truba$time - x_start))]
      t_end_closest   <- w_truba$time[which.min(abs(w_truba$time - x_end))]
      
      w_truba %>%
        filter(time >= t_start_closest,
               time <= t_end_closest) %>%
        summarise(mean_wave = mean(wave, na.rm = TRUE)) %>%
        pull(mean_wave)
    }
  ) %>%
  ungroup()

# separating rise and fall events:

df_slope_env_3h_fall <- df_slope_env_3h[df_slope_env$type=="fall",]
df_slope_env_3h_rise <- df_slope_env_3h[df_slope_env$type=="rise",]

w_truba_slope_rise <- w_truba_slope[w_truba_slope$slope>0,]
w_truba_slope_fall <- w_truba_slope[w_truba_slope$slope<0,]

# creating linear regression models for wind and wave:

windmodel <- lm(slope ~ wind_speed_m_s_wx, data = df_slope_env_3h_rise)
truba_wavemodel <- lm(slope ~ mean_wave, data = w_truba_slope_rise)

##### plotting: ######

# the foundation for graph B:

ggheat <- ggplot(heatsmall, aes(x = time_bin, y = depth, fill = value)) + 
  geom_raster() +
  scale_fill_viridis(discrete = F, option = "mako")+
  theme_minimal(base_size = 7)+
  labs(x= "Date", y= "Depth [m]", fill = c("Echo \nint. [dB]"))+
  theme(panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        #panel.background = element_blank(), 
        axis.text.x = element_text(angle = 45, hjust = 1, colour = "black"),
        axis.ticks.x = element_line(colour = "black"),
        axis.ticks.y = element_line(colour = "black"),
        axis.ticks.length = unit(0.1, "cm"),
        legend.position="right",
        axis.text.y = element_text(colour = "black"))+
  geom_hline(yintercept = c(-9.5, -20.5), linewidth = .5, linetype = "dashed", colour = "gray80") +
  geom_rect(
    aes(
      xmin = as.POSIXct("2018-08-29 14:00:00"),
      xmax = as.POSIXct("2018-08-31 14:00:15"),
      ymin = -20.5,
      ymax = -9.5
    ),
    fill = NA,
    colour = "black",
    linewidth = .5
  )+
  coord_cartesian(expand = F)+
  scale_x_datetime(
    date_breaks = "1 day",
    date_labels = "%d/%m"
  )

# rescale data to be able to create a secondary y-axis:

d_min <- min(sub$depth, na.rm = TRUE)
d_max <- max(sub$depth, na.rm = TRUE)

s_min <- min(df$smooth, na.rm = TRUE)
s_max <- max(df$smooth, na.rm = TRUE)

rescale_to_depth <- function(x) {
  (x - s_min) / (s_max - s_min) * (d_max - d_min) + d_min
}

# apply rescaling to df

df$scaled_smooth <- rescale_to_depth(df$smooth)
results_evening$scaled_smooth <- rescale_to_depth(results_evening$smooth)
results_morning$scaled_smooth <- rescale_to_depth(results_morning$smooth)
df_slopes$y_start_scaled <- rescale_to_depth(df_slopes$y_start)
df_slopes$y_end_scaled   <- rescale_to_depth(df_slopes$y_end)

df_slopes$type <- "Slope"

fill_limits <- range(heatsmall$value, na.rm = T)

# create the cutout plot C:

composite <- ggplot() +
  
  geom_raster(data = sub,
              aes(x = time_bin, y = depth, fill = value)) +
  scale_fill_viridis(discrete = FALSE, option = "mako",
                     limits=fill_limits,
                     oob=scales::squish) +
  labs(y = "Depth (m)", fill = "Backscatter\value", x = "Date") +
  coord_cartesian(
    xlim = ymd_hms(c("2018-08-29 12:00:00",
                     "2018-08-31 12:00:15")),
    expand = FALSE
  ) +
  
  theme_minimal(base_size = 7) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, colour = "black"),
    
    axis.ticks.x = element_line(colour = "black"),
    axis.ticks.length = unit(0.1, "cm"),
    
    axis.text.y.left = element_text(colour = "black"),
    axis.text.y.right = element_text(colour = "black"),
    axis.ticks.y.left = element_line(color = "black"),
    axis.ticks.y.right = element_line(color = "black"),
    legend.key.width = unit(1.5, "cm"),
    axis.title.x = element_blank()
  ) +
  
  scale_x_datetime(
    date_breaks = "1 day",
    date_labels = "%d/%m"
  ) +
  
  ggnewscale::new_scale_colour() +
  
  geom_line(data = df,
            aes(x = time, y = scaled_smooth, colour = "Echo"),
            linewidth = .5
            #,            linetype="dashed"
  ) +
  
  geom_segment(data = df_slopes,
               aes(x = x_start, y = y_start_scaled,
                   xend = x_end, yend = y_end_scaled,
                   colour = type),
               linewidth = .5) +
  
  geom_point(data = results_evening,
             aes(x = evening_onset, y = scaled_smooth,
                 colour = "Ascend"), size = .75) +
  geom_point(data = results_morning,
             aes(x = morning_onset, y = scaled_smooth,
                 colour = "Descend"), size = .75) +
  
  scale_colour_manual(
    name = "",
    values = c(
      Echo = "#bda2b5",
      Ascend = "steelblue1",
      Descend = "brown1",
      Slope  = "yellow"   # New slope color
    ),
    breaks = c("Ascend", "Descend", "Echo", "Slope"),
    labels = c("Ascend", "Descend", "Integ. echo", "Slope")
  ) +
  
  # Secondary axis (undoes the scaling)
  scale_y_continuous(
    name = "Depth [m]",
    sec.axis = sec_axis(
      ~ (.-d_min) / (d_max - d_min) * (s_max - s_min) + s_min,
      name = "Integrated echo intensity"
    )
  )+
  guides(fill = "none") +
  theme(legend.position = c(0.5, 1),
        legend.justification = c(0.5, 1),
        legend.direction = "horizontal",
        legend.background = element_rect(fill = "black", linewidth = 0),
        legend.text = element_text(size = 5, colour = "white"),
        legend.key.height = unit(0.01, "cm"),
        legend.key.width = unit(0.30, "cm"))

composite

# apply rescaling for plot B data too, and finish the plot:

d_min_w <- min(heat2$depth, na.rm = TRUE)
d_max_w <- max(heat2$depth, na.rm = TRUE)

rescale_to_depth_w <- function(x) {
  (x - s_min) / (s_max - s_min) * (d_max_w - d_min_w) + d_min_w
}

df$scaled_smooth_w <- rescale_to_depth(df$smooth)

ggheatv2 <- ggheat+
  geom_line(
    data = df,
    aes(time, scaled_smooth_w),
    color = "#bda2b5",
    inherit.aes = FALSE,
    linewidth = .3
    #,    linetype = "dashed"
  ) +
  theme(plot.margin = margin(0,0,0,0, "cm"))+
  scale_y_continuous(
    limits = c(-31, 0),
    sec.axis = sec_axis(
      ~ rescale(., from = c(-20.5, -9.5), to = range(df$smooth, na.rm = T)),
      name = "Integrated echo intensity",
      breaks = seq(550, 700, 50)
    )
  )

# function for wave and wind regression plots (D, E):

ggplotRegression <- function(fit) {
  
  require(ggplot2)
  
  x_var <- names(fit$model)[2]
  y_var <- names(fit$model)[1]
  
  caption_text <- paste0(
    "Adj. R² = ", signif(summary(fit)$adj.r.squared, 3), "\n",
    "p = ", signif(summary(fit)$coef[2, 4], 1)
  )
  
  x_pos <- max(fit$model[[x_var]], na.rm = TRUE)
  y_pos <- max(fit$model[[y_var]], na.rm = TRUE)
  
  ggplot(fit$model, aes_string(x = x_var, y = y_var)) +
    geom_point(size = 0.5) +
    stat_smooth(method = "lm", col = "steelblue1", linewidth = 0.5) +
    
    annotate("text", x = x_pos, y = y_pos,
             label = caption_text,
             hjust = 1, vjust = 1,  # top-right corner alignment
             size = 6 / .pt) +
    
    theme_minimal(base_size = 7) +
    theme(
      axis.ticks.length = unit(0.1, "cm"),
      axis.text = element_text(#size = 7,
        color = "black"),
      #axis.title = element_text(size = 7)
    )
}

# plots C and D:

ggwind <- ggplotRegression(windmodel)+ 
  labs(x="Wind speed [m/s]", y="Ascent rate")

ggtrubawave <- ggplotRegression(truba_wavemodel)+ 
  labs(x="Sign. wave height [m]", y="Ascent rate")

# plot A with ADCP wave data for the whole duration:

ggwave_demo <- ggplot(data = w_truba, aes(x=time, y=smooth))+
  geom_line(linewidth = 0.2)+
  labs(y="S. wave\nheight [m]")+
  coord_cartesian(expand = F)+
  theme_minimal(base_size = 7)+
  theme(axis.ticks.length = unit(0.1, "cm"),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        panel.grid.minor = element_blank(),
        plot.margin = margin(2,0,0,1, "mm"))+
  scale_x_datetime(date_breaks = "1 day", 
                   date_labels = "%d/%m")+
  scale_y_continuous(position = "right")

ggplot(data = df_slope_env_3h_rise)+
  geom_point(aes(x=datetime_wx, y=slope))

# finally a combined plot for everything:

blank_plot <- ggplot()+
  geom_blank()+
  theme_minimal()

ggsave(plot = 
         ggarrange(
           ggarrange(ggwave_demo, ggheatv2,
                     nrow=2,
                     labels =c("A", "B"),
                     font.label = list(size = 10),
                     heights = c(15,45),
                     align = "v"),
           ggarrange(blank_plot, composite, blank_plot, 
                     ncol = 3, 
                     labels = c("", "C", ""), 
                     align = "h", 
                     widths = c(0.8,3,0.8), 
                     font.label = list(size = 10)),
           ggarrange(blank_plot, ggwind, ggtrubawave, blank_plot,
                     ncol = 4, 
                     labels = c("", "D", "E", ""), 
                     align = "h", 
                     widths = c(0.8,1.5,1.5,0.8), 
                     font.label = list(size = 10)),
           heights = c(60, 45, 50),
           nrow = 3
         ), 
       filename = "combined_ALL_v4.svg",
       height = 164, 
       width = 180, 
       units = "mm")
