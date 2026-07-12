// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// GitHub *project* pages serve under /featurizer/ — `site` + `base` must both
// be set or every internal link 404s on the deployed site (plan phase-1 spike).
export default defineConfig({
	site: 'https://ccd-ia.github.io',
	base: '/featurizer',
	integrations: [
		starlight({
			title: 'featurizer',
			description:
				'Automated feature engineering for temporal data — Deep Feature Synthesis compiled to pure PostgreSQL.',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/ccd-ia/featurizer' },
			],
			customCss: ['./src/styles/custom.css'],
			// Sidebar groups are added phase by phase as their content lands
			// (Starlight fails the build on sidebar slugs without pages).
			sidebar: [
				{
					label: 'Validation',
					items: [
						{
							// NOTE: Starlight prepends `base` to sidebar links
							// automatically; content-markdown links need the
							// /featurizer/ prefix written out. Asymmetric on
							// purpose — check_links.py guards both.
							label: 'Live-DB reports (v0.8.0)',
							link: '/specs/live-db-revalidation-v080.html',
							attrs: { target: '_blank' },
						},
					],
				},
			],
		}),
	],
});
