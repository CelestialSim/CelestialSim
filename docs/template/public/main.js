export default {
    start: () => {
        const inheritedSections = document.querySelectorAll('dl.typelist.inheritedMembers')

        for (const section of inheritedSections) {
            const titleNode = section.querySelector(':scope > dt')
            const contentNode = section.querySelector(':scope > dd')
            if (!titleNode || !contentNode) {
                continue
            }

            const details = document.createElement('details')
            details.className = 'typelist inheritedMembers inheritedMembers-collapsible'

            const summary = document.createElement('summary')
            summary.textContent = titleNode.textContent?.trim() || 'Inherited Members'

            const contentWrapper = document.createElement('div')
            contentWrapper.className = 'inheritedMembersContent'
            while (contentNode.firstChild) {
                contentWrapper.appendChild(contentNode.firstChild)
            }

            details.appendChild(summary)
            details.appendChild(contentWrapper)
            section.replaceWith(details)
        }
    }
}
