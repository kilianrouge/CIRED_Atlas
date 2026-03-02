// ATLAS frontend logic — Alpine.js app
let _condId = 0;

function atlasApp() {
  return {
    // Panel
    panel: 'search',

    // Profiles
    profiles: [],
    activeProfileId: null,
    activeProfile: null,
    selectedProfileIds: [],   // checked for running
    runningProfileIds: [],    // currently being searched (live)

    editConditions: [],
    editCombinator: 'OR',
    conditionsChanged: false,

    // Run state
    runTimeframe: 'since_last_run',
    weightArbitration: 0.3,
    runQueue: [],      // [{run_id, profile_id, profile_name, status, n_results}]
    _pollTimer: null,

    // Results
    results: [],
    selectedResultIds: [],
    resultSort: 'score',

    // History
    historyRuns: [],

    // Auth / multi-user
    currentUser: null,

    // Settings
    settings: {
      zotero_library_id: '',
      zotero_api_key: '',
      zotero_library_type: 'user',
      openalex_email: '',
      default_collection_key: '',
      default_inbox_subcollection: '',
      default_tag: 'atlas-import',
      weight_arbitration: 0.3,
      similarity_threshold: 0,
      max_per_condition: 30,
    },
    flatCollections: [],
    settingSections: { zotero: true, collection: false, scoring: false },

    // Modal
    showProfileModal: false,
    editProfileId: null,
    profileForm: { name: '', description: '' },

    // Import conditions modal
    showImportModal: false,
    importJson: '',
    importError: '',

    // Toasts
    toasts: [],

    // ── Computed: group simultaneously-launched runs by group_id ─────────
    get historyGroups() {
      const map = new Map();
      for (const run of this.historyRuns) {
        const key = run.group_id || `_solo_${run.id}`;
        if (!map.has(key)) {
          map.set(key, { group_id: run.group_id, runs: [], _expanded: false, _results: null });
        }
        map.get(key).runs.push(run);
      }
      return [...map.values()].sort((a, b) => {
        const ta = [...a.runs].sort((x, y) => y.started_at.localeCompare(x.started_at))[0]?.started_at || '';
        const tb = [...b.runs].sort((x, y) => y.started_at.localeCompare(x.started_at))[0]?.started_at || '';
        return tb.localeCompare(ta);
      });
    },

    // ── Init ────────────────────────────────────────────────────────────
    async init() {
      const r = await fetch('/api/whoami');
      const d = await r.json();
      this.currentUser = d.user;
      await this.loadProfiles();
      await this.loadSettings();
    },

    // ── Sidebar title ───────────────────────────────────────────────────
    topbarTitle() {
      if (this.panel === 'history') return 'Past Searches';
      if (this.panel === 'settings') return 'Settings';
      if (this.activeProfile) return this.activeProfile.name;
      return 'ATLAS';
    },

    // ── Profile management ───────────────────────────────────────────────
    async loadProfiles() {
      const r = await fetch('/api/profiles');
      this.profiles = await r.json();
    },

    setActiveProfile(p) {
      this.activeProfileId = p.id;
      this.activeProfile = p;
      const conds = (p.conditions || []).map(c => ({
        ...c, _id: _condId++, _display: c._display || c.value || '',
        _acItems: [], _acOpen: false, _excludeRaw: (c.exclude_title || []).join(', '),
        language: c.language || 'en', doc_type: c.doc_type || 'article',
        scope: c.scope || ['language','doc_type'].includes(c.type),
      }));
      this.editConditions = conds;
      this.editCombinator = p.combinator || 'OR';
      this.conditionsChanged = false;
    },

    toggleProfileSelection(id) {
      const idx = this.selectedProfileIds.indexOf(id);
      if (idx === -1) this.selectedProfileIds.push(id);
      else this.selectedProfileIds.splice(idx, 1);
    },

    openNewProfileModal() {
      this.editProfileId = null;
      this.profileForm = { name: '', description: '' };
      this.showProfileModal = true;
    },

    openEditProfileModal() {
      if (!this.activeProfile) return;
      this.editProfileId = this.activeProfile.id;
      this.profileForm = { name: this.activeProfile.name, description: this.activeProfile.description || '' };
      this.showProfileModal = true;
    },

    async saveProfileModal() {
      if (!this.profileForm.name.trim()) { this.toast('Profile name is required', 'error'); return; }
      const url = this.editProfileId ? `/api/profiles/${this.editProfileId}` : '/api/profiles';
      const method = this.editProfileId ? 'PUT' : 'POST';
      const r = await fetch(url, {
        method, headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: this.profileForm.name.trim(), description: this.profileForm.description })
      });
      const data = await r.json();
      this.showProfileModal = false;
      await this.loadProfiles();
      if (!this.editProfileId) {
        const p = this.profiles.find(x => x.id === data.id);
        if (p) this.setActiveProfile(p);
      } else {
        this.activeProfile = this.profiles.find(x => x.id === this.editProfileId) || this.activeProfile;
      }
      this.toast(this.editProfileId ? 'Profile updated' : 'Profile created');
    },

    async deleteProfile() {
      if (!this.activeProfile) return;
      if (!confirm(`Delete profile "${this.activeProfile.name}"?`)) return;
      await fetch(`/api/profiles/${this.activeProfile.id}`, { method: 'DELETE' });
      this.selectedProfileIds = this.selectedProfileIds.filter(id => id !== this.activeProfile.id);
      this.activeProfileId = null;
      this.activeProfile = null;
      this.editConditions = [];
      await this.loadProfiles();
      this.toast('Profile deleted');
    },

    // ── Conditions ───────────────────────────────────────────────────────
    addCondition() {
      this.editConditions.push({
        _id: _condId++, type: 'keywords_title_abstract',
        value: '', label: '', _display: '', _acItems: [], _acOpen: false, _excludeRaw: '',
        scope: false
      });
      this.conditionsChanged = true;
    },

    removeCondition(idx) {
      this.editConditions.splice(idx, 1);
      this.conditionsChanged = true;
    },

    onCondTypeChange(cond) {
      cond.value = '';
      cond._display = '';
      cond._acItems = [];
      cond.min_papers = cond.min_papers || 3;
      // language and doc_type are always scope conditions
      if (['language', 'doc_type'].includes(cond.type)) cond.scope = true;
      this.conditionsChanged = true;
    },

    toggleScope(cond) {
      cond.scope = !cond.scope;
      this.conditionsChanged = true;
    },

    async saveConditions() {
      if (!this.activeProfile) return;
      const conditions = this.editConditions.map(c => {
        const out = { type: c.type, value: c.value, label: c.label || '' };
        if (c._excludeRaw) out.exclude_title = c._excludeRaw.split(',').map(s => s.trim()).filter(Boolean);
        if (c.type === 'author_in_library') out.min_papers = c.min_papers || 3;
        if (['author','journal','topic'].includes(c.type)) out._display = c._display;
        if (c.scope) out.scope = true;
        return out;
      });
      const r = await fetch(`/api/profiles/${this.activeProfile.id}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conditions, combinator: this.editCombinator })
      });
      const data = await r.json();
      this.conditionsChanged = false;
      await this.loadProfiles();
      this.activeProfile = this.profiles.find(p => p.id === this.activeProfile.id) || this.activeProfile;
      this.toast('Conditions saved');
    },

    // Watch for condition changes
    get _conditionsWatch() {
      return JSON.stringify(this.editConditions) + this.editCombinator;
    },

    // ── Autocomplete ─────────────────────────────────────────────────────
    async autocomplete(type, cond, q) {
      if (!q || q.length < 2) { cond._acItems = []; return; }
      const r = await fetch(`/api/autocomplete/${type}?q=${encodeURIComponent(q)}`);
      const data = await r.json();
      cond._acItems = data.results || [];
      cond._acOpen = true;
    },

    selectAcItem(cond, item) {
      cond.value = item.id;
      cond._display = item.display_name;
      cond._acItems = [];
      cond._acOpen = false;
      this.conditionsChanged = true;
    },

    // ── Import / Export conditions JSON ──────────────────────────────────
    importConditionsModal() {
      this.importJson = '';
      this.importError = '';
      this.showImportModal = true;
    },

    applyImportedConditions() {
      this.importError = '';
      let parsed;
      try {
        parsed = JSON.parse(this.importJson.trim());
      } catch(e) {
        this.importError = 'Invalid JSON: ' + e.message;
        return;
      }
      if (!Array.isArray(parsed)) {
        this.importError = 'Expected a JSON array of condition objects.';
        return;
      }
      const VALID_TYPES = ['keywords_title_abstract','keywords_title','keywords_abstract',
        'author','journal','field','domain','citing_library','author_in_library','language','doc_type'];
      const imported = [];
      for (const c of parsed) {
        if (!c.type || !VALID_TYPES.includes(c.type)) {
          this.importError = `Unknown condition type: "${c.type}". Valid types: ${VALID_TYPES.join(', ')}`;
          return;
        }
        imported.push({
          _id: _condId++,
          type: c.type,
          value: c.value || '',
          label: c.label || '',
          _display: c._display || c.value || '',
          _acItems: [],
          _acOpen: false,
          _excludeRaw: (c.exclude_title || []).join(', '),
          min_papers: c.min_papers || 3,
          scope: c.scope || ['language','doc_type'].includes(c.type),
        });
      }
      this.editConditions = [...this.editConditions, ...imported];
      this.conditionsChanged = true;
      this.showImportModal = false;
      this.toast(`${imported.length} condition${imported.length === 1 ? '' : 's'} imported`);
    },

    exportConditions() {
      const conditions = this.editConditions.map(c => {
        const out = { type: c.type };
        if (c.value) out.value = c.value;
        if (c.label) out.label = c.label;
        if (['author','journal'].includes(c.type) && c._display) out._display = c._display;
        if (c._excludeRaw) out.exclude_title = c._excludeRaw.split(',').map(s => s.trim()).filter(Boolean);
        if (c.type === 'author_in_library' && c.min_papers) out.min_papers = c.min_papers;
        if (c.scope) out.scope = true;
        return out;
      });
      const text = JSON.stringify(conditions, null, 2);
      navigator.clipboard.writeText(text).then(
        () => this.toast('Conditions JSON copied to clipboard'),
        () => {
          // Fallback: show in prompt
          prompt('Copy the conditions JSON:', text);
        }
      );
    },

    // ── Run search ───────────────────────────────────────────────────────
    async startRun() {
      if (!this.selectedProfileIds.length) { this.toast('Select at least one profile', 'error'); return; }
      await this.saveConditions();

      const body = {
        profile_ids: this.selectedProfileIds,
        timeframe: this.runTimeframe,
        weight_arbitration: this.weightArbitration,
        collection_key: this.settings.default_collection_key || null
      };
      const r = await fetch('/api/search/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      });
      if (!r.ok) { const e = await r.json(); this.toast(e.error || 'Error starting run', 'error'); return; }
      const runs = await r.json();

      // Add to run queue
      for (const run of runs) {
        this.runQueue.push({ run_id: run.run_id, profile_id: run.profile_id,
          profile_name: run.profile_name, status: 'running',
          n_results: 0, progress_pct: 0, progress_step: 'Starting…' });
        this.runningProfileIds.push(run.profile_id);
      }
      this.results = [];
      this.selectedResultIds = [];

      // Poll all runs
      this._startPolling();
    },

    _startPolling() {
      if (this._pollTimer) clearInterval(this._pollTimer);
      this._pollTimer = setInterval(() => this._pollAll(), 2500);
    },

    async _pollAll() {
      const running = this.runQueue.filter(r => r.status === 'running');
      if (!running.length) { clearInterval(this._pollTimer); return; }

      await Promise.all(running.map(async (qrun) => {
        try {
          const r = await fetch(`/api/search/run/${qrun.run_id}/status`);
          const data = await r.json();
          qrun.status = data.status;
          qrun.progress_pct  = data.progress_pct  || 0;
          qrun.progress_step = data.progress_step || '';
          if (data.status === 'done' || data.status === 'error') {
            this.runningProfileIds = this.runningProfileIds.filter(id => id !== qrun.profile_id);
            if (data.status === 'done') {
              const rr = await fetch(`/api/search/run/${qrun.run_id}/results`);
              const res = await rr.json();
              qrun.n_results = res.length;
              for (const paper of res) {
                paper._profile_name = qrun.profile_name;
                paper._expanded = false;
                if (!this.results.find(x => x.id === paper.id)) this.results.push(paper);
              }
              this.toast(`${qrun.profile_name}: ${res.length} papers found`);
            } else {
              this.toast(`${qrun.profile_name}: search failed`, 'error');
            }
          }
        } catch(e) { console.error('Poll error', e); }
      }));
    },

    // ── Results ──────────────────────────────────────────────────────────
    get pendingResults() { return this.results.filter(r => r.status === 'pending'); },
    get pendingCount() { return this.pendingResults.length; },

    get sortedResults() {
      const arr = [...this.results];
      if (this.resultSort === 'score') arr.sort((a,b) => (b.combined_score||0)-(a.combined_score||0));
      else if (this.resultSort === 'year_desc') arr.sort((a,b) => (b.year||0)-(a.year||0));
      else if (this.resultSort === 'year_asc') arr.sort((a,b) => (a.year||0)-(b.year||0));
      else if (this.resultSort === 'citations') arr.sort((a,b) => (b.cited_by_count||0)-(a.cited_by_count||0));
      return arr;
    },

    toggleResultSelect(id) {
      const idx = this.selectedResultIds.indexOf(id);
      if (idx === -1) this.selectedResultIds.push(id);
      else this.selectedResultIds.splice(idx, 1);
    },

    toggleSelectAll() {
      if (this.selectedResultIds.length === this.pendingResults.length && this.pendingResults.length > 0) {
        this.selectedResultIds = [];
      } else {
        this.selectedResultIds = this.pendingResults.map(r => r.id);
      }
    },

    async decideSingle(r, decision, addToZotero) {
      const res = await fetch(`/api/results/${r.id}/decide`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, add_to_zotero: addToZotero,
          collection_key: this.settings.default_collection_key,
          inbox_subcollection: this.settings.default_inbox_subcollection,
          tag: this.settings.default_tag })
      });
      const data = await res.json();
      r.status = decision;
      this.selectedResultIds = this.selectedResultIds.filter(id => id !== r.id);
      if (addToZotero && data.zotero_added) this.toast('Added to Zotero');
      else this.toast(decision === 'accepted' ? 'Accepted' : decision === 'rejected' ? 'Rejected' : 'Skipped');
    },

    async batchDecide(decision, addToZotero) {
      if (!this.selectedResultIds.length) return;
      const res = await fetch('/api/results/batch_decide', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result_ids: this.selectedResultIds, decision,
          add_to_zotero: addToZotero,
          collection_key: this.settings.default_collection_key,
          inbox_subcollection: this.settings.default_inbox_subcollection,
          tag: this.settings.default_tag })
      });
      const data = await res.json();
      for (const id of this.selectedResultIds) {
        const r = this.results.find(x => x.id === id);
        if (r) r.status = decision;
      }
      const n = this.selectedResultIds.length;
      this.selectedResultIds = [];
      this.toast(`${n} papers ${decision}` + (addToZotero ? ' — pushed to Zotero' : ''));
    },

    async addToZoteroRetro(r) {
      await fetch(`/api/papers/${r.paper_oa_id}/add_to_zotero`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ collection_key: this.settings.default_collection_key,
          inbox_subcollection: this.settings.default_inbox_subcollection, tag: this.settings.default_tag })
      });
      this.toast('Added to Zotero');
    },

    copyDoi(doi) {
      if (!doi) return;
      navigator.clipboard.writeText('https://doi.org/' + doi).then(() => this.toast('DOI copied'));
    },

    // ── Auth / multi-user ─────────────────────────────────────────────────────
    // (user is set server-side from the Unix $USER env var — no login needed)

    // ── History ────────────────────────────────────────────────────────────
    async loadHistory() {
      const r = await fetch('/api/history');
      this.historyRuns = await r.json();
    },

    async toggleHistoryGroup(group) {
      group._expanded = !group._expanded;
      if (group._expanded && !group._results) {
        const allResults = [];
        for (const run of group.runs) {
          const r = await fetch(`/api/search/run/${run.id}/results`);
          const papers = (await r.json()).map(x => ({ ...x, _expanded: false, _profile_name: run.profile_name }));
          allResults.push(...papers);
        }
        // Merge across runs: deduplicate by paper_oa_id, keep highest score
        const best = new Map();
        for (const p of allResults) {
          if (!best.has(p.paper_oa_id) || (p.combined_score || 0) > (best.get(p.paper_oa_id).combined_score || 0))
            best.set(p.paper_oa_id, p);
        }
        group._results = [...best.values()].sort((a, b) => (b.combined_score || 0) - (a.combined_score || 0));
      }
    },

    async deleteHistoryGroup(group, $event) {
      if ($event) $event.stopPropagation();
      const names = group.runs.map(r => r.profile_name || 'Unnamed').join(', ');
      if (!confirm(`Remove “${names}” from history?\n\nThis also removes its decisions from training memory, allowing those papers to appear again in future searches.`)) return;
      for (const run of group.runs) {
        await fetch(`/api/search/run/${run.id}`, { method: 'DELETE' });
      }
      const ids = new Set(group.runs.map(r => r.id));
      this.historyRuns = this.historyRuns.filter(r => !ids.has(r.id));
      this.toast('Search removed from history.');
    },

    async addAllAcceptedToZotero(group) {
      const accepted = (group._results || []).filter(r => r.status === 'accepted');
      for (const r of accepted) await this.addToZoteroRetro(r);
      this.toast(`Pushed ${accepted.length} papers to Zotero`);
    },

    // ── Settings ─────────────────────────────────────────────────────────
    async loadSettings() {
      const r = await fetch('/api/settings');
      const data = await r.json();
      this.settings = { ...this.settings, ...data };
      if (data.weight_arbitration !== undefined) this.weightArbitration = +data.weight_arbitration;
    },

    async saveSettings() {
      const r = await fetch('/api/settings', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.settings)
      });
      if (r.ok) this.toast('Settings saved');
      else this.toast('Save failed', 'error');
    },

    async loadZoteroCollections() {
      const r = await fetch('/api/zotero/collections/flat');
      if (!r.ok) { this.toast('Failed to load Zotero collections', 'error'); return; }
      this.flatCollections = await r.json();
      this.toast(`${this.flatCollections.length} collections loaded`);
    },

    // ── Helpers ──────────────────────────────────────────────────────────
    formatAuthors(authors, max) {
      if (!authors || !authors.length) return 'Unknown';
      const names = authors.map(a => typeof a === 'string' ? a : (a.name || a.display_name || ''));
      if (names.length <= max) return names.join(', ');
      return names.slice(0, max).join(', ') + ' et al.';
    },

    formatDate(dt) {
      if (!dt) return '?';
      const d = new Date(dt);
      return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
    },

    // ── Toasts ───────────────────────────────────────────────────────────
    toast(msg, type = 'info') {
      const id = Date.now() + Math.random();
      this.toasts.push({ id, msg, type });
      setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 3500);
    },
  };
}
