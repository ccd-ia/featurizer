// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';

// GitHub *project* pages serve under /featurizer/ — `site` + `base` must both
// be set or every internal link 404s on the deployed site (plan phase-1 spike).
export default defineConfig({
	site: 'https://ccd-ia.github.io',
	base: '/featurizer',
	// MathJax rendered at BUILD time to inline SVG (rehype-mathjax): no CDN
	// script at runtime, pages stay self-contained. $...$ / $$...$$ in markdown.
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeMathjax],
	},
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
					label: 'Start Here',
					items: [{ label: 'Walkthrough', slug: 'walkthrough' }],
				},
				{
					label: 'Concepts',
					items: [{ autogenerate: { directory: 'concepts' } }],
				},
				{
					label: 'Notebooks',
					items: [{ autogenerate: { directory: 'notebooks' } }],
				},
				{
					label: 'Reference',
					items: [{ autogenerate: { directory: 'reference' } }],
				},
				{
					label: 'Engineering',
					items: [
						{ label: 'Performance internals', slug: 'engineering/internals' },
						{ label: 'Architecture decisions', slug: 'engineering/adr' },
						{ label: 'Changelog', slug: 'engineering/changelog' },
					],
				},
				{
					label: 'Validation',
					items: [
						{
							// Full URL on purpose: Starlight normalizes internal
							// sidebar links to extensionless routes (strips .html),
							// which 404s for pass-through static artifacts. External
							// URLs pass through untouched. Content-markdown links to
							// artifacts need raw <a href> for the same reason.
							label: 'Live-DB reports (v0.8.0)',
							link: 'https://ccd-ia.github.io/featurizer/specs/live-db-revalidation-v080.html',
							attrs: { target: '_blank' },
						},
					],
				},
			],
		}),
	],
});
