(() => {
  const d = window.CHURN_DASHBOARD || {};
  const m = d.metrics || {};
  const palette = ['#6c5ce7','#4d9de0','#f4a259','#2bc48a','#b29cff','#ff7c91','#6c7a89','#e0b34f'];
  const fmt = v => Number(v || 0);
  const metricLabels = ['Accuracy','Balanced acc.','Precision','Recall','F1','ROC-AUC','Avg. precision','MCC'];
  const metricValues = [
    fmt(m['Accuracy']),fmt(m['Balanced accuracy']),fmt(m['Precision (churn)']),
    fmt(m['Recall (churn)']),fmt(m['F1 (churn)']),fmt(m['ROC-AUC']),
    fmt(m['Average precision']),Math.max(0,(fmt(m['MCC'])+1)/2)
  ];
  const ctx = document.getElementById('metricsChart');
  if (ctx && window.Chart) {
    new Chart(ctx,{type:'bar',data:{labels:metricLabels,datasets:[{label:'Score',data:metricValues,backgroundColor:palette,borderRadius:7,maxBarThickness:24}]},
      options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.raw.toFixed(3)}`}}},
        scales:{x:{min:0,max:1,grid:{color:'#f0eff6'},ticks:{font:{size:10}}},y:{grid:{display:false},ticks:{font:{size:10}}}}}});
  }
  const comparison = d.comparison || [];
  const compCtx = document.getElementById('comparisonChart');
  if (compCtx && window.Chart) {
    new Chart(compCtx,{type:'bar',data:{labels:comparison.map(x=>x.Model || x.model || 'Model'),datasets:[
      {label:'ROC-AUC',data:comparison.map(x=>fmt(x['ROC-AUC'])),backgroundColor:'#6c5ce7',borderRadius:5},
      {label:'Average precision',data:comparison.map(x=>fmt(x['Average precision'] ?? x['PR-AUC (avg precision)'])),backgroundColor:'#a89bff',borderRadius:5}
    ]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{boxWidth:9,font:{size:10}}}},
      scales:{y:{min:0,max:1,grid:{color:'#f0eff6'},ticks:{font:{size:10}}},x:{grid:{display:false},ticks:{font:{size:9},maxRotation:35,minRotation:0}}}}});
  }
  const importance = d.importance || [];
  const impCtx = document.getElementById('importanceChart');
  if (impCtx && window.Chart && importance.length) {
    const top = importance.slice(0,8).reverse();
    new Chart(impCtx,{type:'bar',data:{labels:top.map(x=>x.feature),datasets:[{label:'Importance',data:top.map(x=>fmt(x.importance_mean)),backgroundColor:'#4fbd9b',borderRadius:5}]},
      options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`Δ average precision: ${c.raw.toFixed(4)}`}}},
        scales:{x:{grid:{color:'#f0eff6'},ticks:{font:{size:9}}},y:{grid:{display:false},ticks:{font:{size:9}}}}}});
  } else if (impCtx) {
    impCtx.parentElement.innerHTML='<p class="muted small">Feature importance artifact not available yet. Run the notebook to generate it.</p>';
  }
})();
