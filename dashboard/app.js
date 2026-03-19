const COLORS = ['#8b6cff', '#34d399', '#fb923c', '#f87171', '#60a5fa', '#e879f9', '#facc15', '#2dd4bf'];

function formatUptime(seconds) {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const parts = [];
  if (d > 0) parts.push(d + 'd');
  if (h > 0) parts.push(h + 'h');
  if (m > 0) parts.push(m + 'm');
  parts.push(s + 's');
  return 'Up ' + parts.join(' ');
}

function formatNumber(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
  return n.toLocaleString();
}

function formatCost(n) {
  if (n >= 1) return '$' + n.toFixed(2);
  if (n >= 0.01) return '$' + n.toFixed(3);
  return '$' + n.toFixed(4);
}

function animateValue(element, newText) {
  if (element.textContent !== newText) {
    element.textContent = newText;
    element.style.transition = 'none';
    element.style.opacity = '0.6';
    requestAnimationFrame(() => {
      element.style.transition = 'opacity 0.3s ease';
      element.style.opacity = '1';
    });
  }
}

async function refresh() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    const data = await res.json();

    animateValue(document.getElementById('total-requests'), formatNumber(data.total_requests));
    animateValue(document.getElementById('total-cost'), formatCost(data.total_cost_usd));
    animateValue(document.getElementById('cache-hits'), formatNumber(data.cache_hits));
    animateValue(document.getElementById('total-errors'), formatNumber(data.total_errors));
    animateValue(document.getElementById('uptime'), formatUptime(data.uptime_seconds));

    if (data.total_requests > 0) {
      const rate = ((data.cache_hits / data.total_requests) * 100).toFixed(1);
      document.getElementById('cache-rate').textContent = rate + '% hit rate';
    }

    // Provider table
    const tbody = document.getElementById('providers-table');
    const providers = Object.entries(data.providers || {});

    if (providers.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No requests yet</td></tr>';
    } else {
      tbody.innerHTML = providers.map(function(entry) {
        const name = entry[0];
        const s = entry[1];
        return '<tr>' +
          '<td><span class="provider-name">' + name + '</span></td>' +
          '<td class="mono">' + formatNumber(s.requests) + '</td>' +
          '<td class="mono">' + formatNumber(s.input_tokens) + '</td>' +
          '<td class="mono">' + formatNumber(s.output_tokens) + '</td>' +
          '<td class="mono" style="color: var(--green)">' + formatCost(s.total_cost_usd) + '</td>' +
          '<td class="mono" style="color: ' + (s.errors > 0 ? 'var(--red)' : 'var(--muted)') + '">' + s.errors + '</td>' +
          '</tr>';
      }).join('');
    }

    // Usage distribution bars
    var barContainer = document.getElementById('usage-bars');
    var barRows = document.getElementById('bar-rows');

    if (providers.length > 0) {
      barContainer.style.display = 'block';
      var total = providers.reduce(function(sum, entry) { return sum + entry[1].requests; }, 0);

      barRows.innerHTML = providers.map(function(entry, i) {
        var name = entry[0];
        var s = entry[1];
        var pct = total > 0 ? (s.requests / total * 100) : 0;
        var color = COLORS[i % COLORS.length];
        return '<div class="bar-row">' +
          '<div class="bar-label">' + name + '</div>' +
          '<div class="bar-track">' +
          '<div class="bar-fill" style="width: ' + pct + '%; background: ' + color + '"></div>' +
          '</div>' +
          '<div class="bar-value">' + pct.toFixed(1) + '%</div>' +
          '</div>';
      }).join('');
    }
  } catch (e) {
    console.error('Failed to fetch stats:', e);
  }
}

// Initial load and auto-refresh
refresh();
setInterval(refresh, 2000);
