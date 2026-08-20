# Explicit base-R reference for System GMM on Wooldridge wagepan.
#
# This intentionally mirrors Greeners' SystemGmm::fit contract instead of using
# plm::pgmm, whose default instrument and weighting conventions differ.

library(wooldridge)
library(jsonlite)

build_system_gmm <- function(y, X, entity_ids, time_ids, max_lags) {
  n_total <- length(y)
  k_x <- ncol(X)

  ord <- order(entity_ids, time_ids)
  ys <- y[ord]
  xs <- X[ord, , drop = FALSE]
  ids <- entity_ids[ord]

  boundaries <- c(1, which(diff(ids) != 0) + 1, n_total + 1)

  dy_vec <- numeric()
  dyl_vec <- numeric()
  dx_rows <- matrix(numeric(), nrow = 0, ncol = k_x)
  zinst_fd <- matrix(numeric(), nrow = 0, ncol = max_lags)

  y_lev <- numeric()
  yl_lev <- numeric()
  x_lev <- matrix(numeric(), nrow = 0, ncol = k_x)
  zinst_lv_base <- numeric()
  dx_lv_rows <- matrix(numeric(), nrow = 0, ncol = k_x)

  entity_fd_count <- integer()
  entity_lev_count <- integer()

  for (b in seq_len(length(boundaries) - 1)) {
    s_start <- boundaries[b]
    s_end <- boundaries[b + 1] - 1
    idx <- s_start:s_end
    t_i <- length(idx)

    if (t_i < 3) {
      entity_fd_count <- c(entity_fd_count, 0L)
      entity_lev_count <- c(entity_lev_count, 0L)
      next
    }

    for (j in 3:t_i) {
      current <- idx[j]
      previous <- idx[j - 1]
      previous2 <- idx[j - 2]

      dy_vec <- c(dy_vec, ys[current] - ys[previous])
      dyl_vec <- c(dyl_vec, ys[previous] - ys[previous2])
      dx_current <- xs[current, ] - xs[previous, ]
      dx_rows <- rbind(dx_rows, dx_current)

      inst <- numeric(max_lags)
      for (l in seq_len(max_lags)) {
        lag <- l + 1
        inst[l] <- if (j > lag) ys[idx[j - lag]] else 0.0
      }
      zinst_fd <- rbind(zinst_fd, inst)

      y_lev <- c(y_lev, ys[current])
      yl_lev <- c(yl_lev, ys[previous])
      x_lev <- rbind(x_lev, xs[current, ])
      zinst_lv_base <- c(zinst_lv_base, ys[previous] - ys[previous2])
      dx_lv_rows <- rbind(dx_lv_rows, xs[previous, ] - xs[previous2, ])
    }

    entity_fd_count <- c(entity_fd_count, t_i - 2L)
    entity_lev_count <- c(entity_lev_count, t_i - 2L)
  }

  n_fd <- length(dy_vec)
  n_lev <- length(y_lev)
  n_sys <- n_fd + n_lev

  active_x <- which(colSums(abs(dx_rows) > 1e-12) > 0)
  k_dx <- length(active_x)
  k_reg <- 1L + k_dx

  n_inst_fd <- max_lags + k_dx
  n_inst_lv <- 1L + k_dx
  n_inst_sys <- n_inst_fd + n_inst_lv

  w_sys <- matrix(0.0, nrow = n_sys, ncol = k_reg)
  z_sys <- matrix(0.0, nrow = n_sys, ncol = n_inst_sys)

  for (i in seq_len(n_fd)) {
    w_sys[i, 1] <- dyl_vec[i]
    for (nc in seq_along(active_x)) {
      oc <- active_x[nc]
      w_sys[i, 1 + nc] <- dx_rows[i, oc]
      z_sys[i, max_lags + nc] <- dx_rows[i, oc]
    }
    z_sys[i, seq_len(max_lags)] <- zinst_fd[i, ]
  }

  for (i in seq_len(n_lev)) {
    row <- n_fd + i
    w_sys[row, 1] <- yl_lev[i]
    for (nc in seq_along(active_x)) {
      oc <- active_x[nc]
      w_sys[row, 1 + nc] <- x_lev[i, oc]
      z_sys[row, n_inst_fd + 1 + nc] <- dx_lv_rows[i, oc]
    }
    z_sys[row, n_inst_fd + 1] <- zinst_lv_base[i]
  }

  zthz <- matrix(0.0, nrow = n_inst_sys, ncol = n_inst_sys)
  rptr_fd <- 1L
  rptr_lev <- n_fd + 1L

  for (idx_entity in seq_along(entity_fd_count)) {
    fc_fd <- entity_fd_count[idx_entity]
    fc_lev <- entity_lev_count[idx_entity]
    if (fc_fd == 0L) {
      next
    }

    zfd <- z_sys[rptr_fd:(rptr_fd + fc_fd - 1L), , drop = FALSE]
    h_fd <- matrix(0.0, nrow = fc_fd, ncol = fc_fd)
    diag(h_fd) <- 2.0
    if (fc_fd > 1L) {
      for (s in 1:(fc_fd - 1L)) {
        h_fd[s, s + 1L] <- -1.0
        h_fd[s + 1L, s] <- -1.0
      }
    }
    zthz <- zthz + t(zfd) %*% h_fd %*% zfd

    zlv <- z_sys[rptr_lev:(rptr_lev + fc_lev - 1L), , drop = FALSE]
    zthz <- zthz + t(zlv) %*% zlv

    rptr_fd <- rptr_fd + fc_fd
    rptr_lev <- rptr_lev + fc_lev
  }

  a1 <- solve(zthz)
  dy_sys <- c(dy_vec, y_lev)
  wtz <- t(w_sys) %*% z_sys
  zty <- t(z_sys) %*% dy_sys
  wtz_a1 <- wtz %*% a1
  lhs1 <- wtz_a1 %*% t(wtz)
  lhs1_inv <- solve(lhs1)
  params1 <- as.vector(lhs1_inv %*% wtz_a1 %*% zty)
  resid1 <- as.vector(dy_sys - w_sys %*% params1)

  sigma <- matrix(0.0, nrow = n_inst_sys, ncol = n_inst_sys)
  rfd <- 1L
  rlev <- n_fd + 1L
  for (idx_entity in seq_along(entity_fd_count)) {
    fc_fd <- entity_fd_count[idx_entity]
    fc_lev <- entity_lev_count[idx_entity]
    if (fc_fd == 0L) {
      next
    }

    z_ent <- rbind(
      z_sys[rfd:(rfd + fc_fd - 1L), , drop = FALSE],
      z_sys[rlev:(rlev + fc_lev - 1L), , drop = FALSE]
    )
    u_ent <- c(
      resid1[rfd:(rfd + fc_fd - 1L)],
      resid1[rlev:(rlev + fc_lev - 1L)]
    )
    zu <- as.vector(t(z_ent) %*% u_ent)
    sigma <- sigma + tcrossprod(zu)

    rfd <- rfd + fc_fd
    rlev <- rlev + fc_lev
  }

  a2 <- solve(sigma)
  wtz_a2 <- wtz %*% a2
  lhs2 <- wtz_a2 %*% t(wtz)
  lhs2_inv <- solve(lhs2)
  params2 <- as.vector(lhs2_inv %*% wtz_a2 %*% zty)
  std_errors <- sqrt(pmax(diag(lhs2_inv), 0.0))

  list(params = params2, std_errors = std_errors)
}

case_dir <- "validation/cases/sysgmm_wagepan"
data_dir <- file.path(case_dir, "data")
ref_dir <- file.path(case_dir, "reference")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)

data(wagepan)
wagepan <- wagepan[order(wagepan$nr, wagepan$year), ]
write.csv(wagepan, file.path(data_dir, "wagepan.csv"), row.names = FALSE)

y <- as.numeric(wagepan$lwage)
X <- as.matrix(wagepan[, c("exper", "expersq", "married", "union")])
entity_ids <- as.integer(wagepan$nr)
time_ids <- as.integer(wagepan$year)

fit <- build_system_gmm(y, X, entity_ids, time_ids, max_lags = 2L)
names <- c("lwage_lag", "exper", "expersq", "married", "union")

result <- list(
  coefficients = as.list(setNames(as.numeric(fit$params), names)),
  standard_errors = as.list(setNames(as.numeric(fit$std_errors), names))
)

write_json(result, file.path(ref_dir, "expected.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
