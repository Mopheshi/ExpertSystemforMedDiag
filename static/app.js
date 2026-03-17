document.addEventListener('DOMContentLoaded', () => {
    const diagnoseBtn = document.getElementById('diagnose-btn');
    const complaintInput = document.getElementById('complaint');
    const resultsContainer = document.getElementById('results-container');
    const mappedContainer = document.getElementById('mapped-symptoms-container');
    const mappedList = document.getElementById('mapped-symptoms-list');
    const unmappedContainer = document.querySelector('#unmapped-warning');
    const unmappedList = document.querySelector('#unmapped-list');
    const networkContainer = document.getElementById('network-container');

    let network = null;

    diagnoseBtn.addEventListener('click', async () => {
        const text = complaintInput.value.trim();
        if (!text) return;

        // UI State: Loading
        setLoadingState(true);
        resultsContainer.innerHTML = '';
        mappedList.innerHTML = '';
        mappedContainer.classList.add('hidden');
        unmappedContainer.classList.add('hidden');

        try {
            const response = await fetch('/api/diagnose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ complaint: text })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Diagnostic failed');
            }

            const data = await response.json();
            
            // 1. Render Mapped Symptoms
            renderMappedSymptoms(data.mapped_symptoms);

            // 2. Render Diagnosis Panel
            renderDiagnosis(data.diagnosis);

            // 3. Render Unmapped Symptoms
            if (data.unmapped_symptoms && data.unmapped_symptoms.length > 0) {
                unmappedList.textContent = data.unmapped_symptoms.join(', ').replace(/_/g, ' ');
                unmappedContainer.classList.remove('hidden');
            }

            // 4. Render DAG (with full structure)
            renderDAG(data.audit_trail, data.mapped_symptoms, data.diagnosis, data.kb_structure);

        } catch (error) {
            console.error(error);
            resultsContainer.innerHTML = `
                <div class="p-4 rounded-md bg-red-50 border border-red-200 text-red-700 text-sm">
                    <strong>Error:</strong> ${error.message}
                </div>
            `;
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        const spinner = diagnoseBtn.querySelector('.animate-spin');
        const arrow = diagnoseBtn.querySelector('svg:not(.animate-spin)');
        diagnoseBtn.disabled = isLoading;
        
        if (isLoading) {
            spinner.classList.remove('hidden');
            arrow.classList.add('hidden');
        } else {
            spinner.classList.add('hidden');
            arrow.classList.remove('hidden');
        }
    }

    function renderMappedSymptoms(mappedSymptoms) {
        if (!mappedSymptoms || Object.keys(mappedSymptoms).length === 0) {
            mappedContainer.classList.add('hidden');
            return;
        }
        
        // Clear previous
        mappedList.innerHTML = '';
        mappedContainer.classList.remove('hidden');

        Object.entries(mappedSymptoms).forEach(([symptom, cf]) => {
            const label = symptom.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            const tag = document.createElement('div');
            tag.className = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800";
            tag.innerHTML = `
                ${label}
                <span class="ml-1.5 px-1 bg-white rounded-full text-[10px] opacity-75 text-blue-600 font-bold">${cf.toFixed(2)}</span>
            `;
            mappedList.appendChild(tag);
        });
    }

    function renderDiagnosis(diagnosis) {
        if (Object.keys(diagnosis).length === 0) {
            resultsContainer.innerHTML = '<div class="text-slate-500 text-sm italic">No diseases matched the symptoms provided.</div>';
            return;
        }

        // The API returns sorted dict, but JS object order isn't guaranteed (though usually preserved for non-integer keys).
        // Let's sort explicitly just in case.
        const sorted = Object.entries(diagnosis).sort(([,a], [,b]) => b - a);

        let html = '';
        sorted.forEach(([disease, cf], index) => {
            // Highlight topmost result
            const isTop = index === 0;
            const bgColor = isTop ? 'bg-white shadow-md border-slate-200' : 'bg-slate-50 border-transparent opacity-80';
            const textColor = isTop ? 'text-slate-900' : 'text-slate-600';
            const barColor = isTop ? 'bg-slate-800' : 'bg-slate-400';
            
            const readableName = disease.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            const percent = Math.round(cf * 100);

            html += `
                <div class="flex items-center p-4 rounded-lg border ${bgColor} transition-all hover:shadow-lg hover:border-slate-300 group cursor-default">
                    <div class="flex-1">
                        <h4 class="text-sm font-semibold ${textColor}">${readableName}</h4>
                        <div class="w-full bg-slate-100 rounded-full h-1.5 mt-2 overflow-hidden">
                            <div class="${barColor} h-1.5 rounded-full" style="width: ${percent}%"></div>
                        </div>
                    </div>
                    <div class="ml-4 text-right">
                        <span class="block text-xl font-bold ${textColor}">${cf.toFixed(2)}</span>
                        <span class="text-[10px] text-slate-400 uppercase tracking-widest font-medium">CF</span>
                    </div>
                </div>
            `;
        });
        resultsContainer.innerHTML = html;
    }

    function renderDAG(auditTrail, mappedSymptoms, finalDiagnosis, kbStructure) {
        // Construct Nodes and Edges using Vis.js format
        const nodes = new vis.DataSet();
        const edges = new vis.DataSet();
        
        // Track IDs
        const existingNodeIds = new Set();
        
        // --- 1. Draw the Full Disease Structure (Sinks) ---
        // Ensure ALL diseases from KB are present, even if 0.0 CF
        const allDiseases = kbStructure ? kbStructure.diseases : Object.keys(finalDiagnosis); 
        
        allDiseases.forEach(disease => {
            const finalCF = finalDiagnosis[disease] || 0.0;
            const isDiagnosed = finalCF > 0.0;
            const label = disease.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            
            nodes.add({
                id: `dis_${disease}`,
                label: `${label}\n(CF: ${finalCF.toFixed(2)})`,
                group: isDiagnosed ? 'disease_active' : 'disease_inactive',
                value: isDiagnosed ? 10 + (finalCF * 10) : 5, // Size scales with CF
                title: `Disease: ${label}`
            });
            existingNodeIds.add(`dis_${disease}`);
        });

        // --- 2. Draw Symptoms and Rules from Static KB Structure (if available) ---
        // If kbStructure is available, we draw the "potential" graph in grey
        if (kbStructure && kbStructure.rules) {
            // Map rules to create edges
            kbStructure.rules.forEach(rule => {
                const diseaseId = `dis_${rule.hypothesis}`;
                const ruleId = `rule_${rule.id}`; // Optional: Could make rule nodes
                
                // For direct S -> D edges approach (cleaner for medical users):
                // We draw edges from Symptom -> Disease
                
                rule.conditions.forEach(condition => {
                    const symptom = condition.symptom;
                    const symptomId = `sym_${symptom}`;
                    
                    // Add Symptom Node if missing (default inactive)
                    if (!existingNodeIds.has(symptomId)) {
                        const isPresent = mappedSymptoms.hasOwnProperty(symptom);
                        const symLabel = symptom.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                        
                        nodes.add({
                            id: symptomId,
                            label: symLabel,
                            group: isPresent ? 'symptom_active' : 'symptom_inactive',
                            title: isPresent ? `Extracted (CF: ${mappedSymptoms[symptom]})` : 'Not Reported'
                        });
                        existingNodeIds.add(symptomId);
                    }
                    
                    // Add "Potential" Edge (Grey)
                    // We only add this if we want to show the full map. 
                    // To avoid clutter, maybe only show edges for PRESENT symptoms?
                    // User asked for "connection between the deseases".
                    // Showing ALL edges makes it a spiderweb. 
                    // Compromise: Show edges for ALL symptoms related to the top diseases?
                    // Or follow the prompt "show connections". 
                    // Let's add all potential edges but transparent/grey.
                    
                    const edgeId = `edge_${symptom}_${rule.hypothesis}_${rule.id}`;
                    edges.add({
                        id: edgeId,
                        from: symptomId,
                        to: diseaseId,
                        arrows: 'to',
                        color: { color: '#e2e8f0', opacity: 0.3 }, // Very faint grey
                        dashes: true
                    });
                });
            });
        }

        // --- 3. Highlight/Override Active Paths from Audit Trail ---
        // These are the rules that ACTUALLY fired.
        auditTrail.forEach((entry, index) => {
            const disease = entry.hypothesis;
            const ruleStrength = entry.series_cf;
            
            Object.keys(entry.matched_symptoms).forEach(symptom => {
                const symptomId = `sym_${symptom}`;
                const diseaseId = `dis_${disease}`;
                
                // Ensure nodes match active state (might already be set, but force it)
                if (existingNodeIds.has(symptomId)) {
                    nodes.update({ id: symptomId, group: 'symptom_active' });
                } else {
                     // Should exist from KB structure, but if specific KB structure missing...
                     const symLabel = symptom.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                     nodes.add({
                        id: symptomId,
                        label: `${symLabel}`,
                        group: 'symptom_active'
                    });
                    existingNodeIds.add(symptomId);
                }
                
                nodes.update({ id: diseaseId, group: 'disease_active' });

                // Create or Update Edge to be visible/solid
                // We find the matching edge or create a new "active" one
                // Since we iterate audit trail, these are definite hits.
                
                // We'll just add a solid edge ON TOP or rely on Vis to handle.
                // Better: Add a distinct active edge.
                edges.add({
                    id: `active_edge_${symptom}_${disease}_${index}`,
                    from: symptomId,
                    to: diseaseId,
                    label: `${ruleStrength.toFixed(2)}`,
                    arrows: 'to',
                    color: { color: '#3b82f6', opacity: 1.0, inherit: false }, // Bright Blue
                    width: 2,
                    font: { align: 'middle', size: 10, background: 'white' }
                });
            });
        });

        // Configuration for Vis.js
        const options = {
            nodes: {
                shape: 'dot',
                font: {
                    face: 'Inter',
                    size: 14,
                    color: '#1e293b'
                },
                borderWidth: 2,
                shadow: true
            },
            groups: {
                symptom_active: {
                    color: { background: '#dbeafe', border: '#2563eb' }, // Blue-50/600
                    shape: 'dot',
                    size: 15
                },
                symptom_inactive: {
                    color: { background: '#f1f5f9', border: '#cbd5e1' }, // Slate-100/300
                    shape: 'dot',
                    size: 10,
                    font: { color: '#94a3b8' } // Grey text
                },
                disease_active: {
                    color: { background: '#1e293b', border: '#0f172a' }, // Slate-800/900
                    font: { color: '#ffffff' },
                    shape: 'box', 
                    margin: 10,
                    size: 25
                },
                disease_inactive: {
                    color: { background: '#e2e8f0', border: '#94a3b8' }, // Slate-200/400
                    font: { color: '#64748b' },
                    shape: 'box',
                    margin: 10,
                    size: 20
                }
            },
            edges: {
                smooth: { type: 'cubicBezier', forceDirection: 'horizontal' }
            },
            layout: {
                hierarchical: {
                    direction: 'LR',
                    sortMethod: 'directed',
                    levelSeparation: 250,
                    nodeSpacing: 100
                }
            },
            physics: {
                hierarchicalRepulsion: {
                    nodeDistance: 120
                },
                stabilization: true
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                zoomView: true
            }
        };

        if (network) {
            network.destroy();
        }
        
        network = new vis.Network(networkContainer, { nodes, edges }, options);
    }
});
